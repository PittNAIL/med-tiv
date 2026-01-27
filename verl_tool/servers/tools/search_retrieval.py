"""
Search Retrieval Tool for verl-tool
- Features: Load Balancing, Caching, Retry Logic
- Thread-Safe: Fixed for high-concurrency environments
- Logging: Immediate writes with proper synchronization
"""
from .base import BaseTool, register_tool
import regex as re
import requests
from typing import Tuple, Dict, Any, List
import logging
import os
from datetime import datetime
from threading import Lock
import threading
import hashlib
import time
from collections import OrderedDict

# Configure logger for this module
logger = logging.getLogger(__name__)

def setup_debug_logger():
    """Setup a separate file handler for detailed debug logging"""
    debug_log_dir = '/ocean/projects/med230010p/hzhang38/labsage/verl-tool/examples/train/search_r1'
    os.makedirs(debug_log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(debug_log_dir, f'search_retrieval_debug_{timestamp}.log')
    
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    
    # Force line buffering for immediate writes
    try:
        file_handler.stream.reconfigure(line_buffering=True)
    except AttributeError:
        # Fallback for older Python versions
        file_handler.stream = os.fdopen(file_handler.stream.fileno(), 'a', 1)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)
    
    # Force a write immediately to verify the file is created
    logger.info(f"Log file created successfully at {log_file}")
    for handler in logger.handlers:
        handler.flush()
    
    return log_file

@register_tool
class SearchRetrievalTool(BaseTool):
    tool_type = "search_retrieval"
    
    def __init__(self, num_workers=1, retriever_url="http://127.0.0.1:8000", topk=3, **kwargs):
        super().__init__(num_workers)
        
        # --- 1. Single Server Setup (Multi-GPU on server side) ---
        self.retriever_url = kwargs.get('retriever_url', os.getenv('RETRIEVER_URL', retriever_url))
        self.topk = kwargs.get('topk', int(os.getenv('RETRIEVER_TOPK', str(topk))))
        
        # --- 2. Thread-Safe Caching Setup (Aggressive for single server) ---
        self.query_cache = OrderedDict()
        self.cache_lock = Lock()
        # Larger cache for single-server setup to maximize hit rate
        self.max_cache_size = int(os.getenv('RETRIEVER_CACHE_SIZE', '50000'))
        
        # --- 3. Thread-Safe Statistics ---
        self.stats_lock = Lock()
        self.total_searches = 0
        self.empty_searches = 0
        self.failed_searches = 0
        self.timeout_count = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.retry_count = 0  # Track retries
        
        # --- 4. Thread-Local Session Storage ---
        self._local = threading.local()
        
        # --- 5. Logging Setup ---
        self.debug_log_file = setup_debug_logger()
        
        # Empty Query Log with synchronized writes
        self.empty_queries_file = self.debug_log_file.replace('.log', '_empty_queries.txt')
        self.empty_log_lock = Lock()
        with open(self.empty_queries_file, 'w') as f:
            f.write(f"Empty Query Log - Started at {datetime.now()}\n")
            f.write("=" * 80 + "\n\n")
            f.flush()
        
        logger.info(f"Initialized with single multi-GPU server: {self.retriever_url}")
        logger.info(f"Cache size: {self.max_cache_size} queries")
        logger.info(f"Logs saving to: {self.debug_log_file}")
        self._flush_logs()

    def _get_session(self):
        """Get thread-local session for thread-safe requests"""
        if not hasattr(self._local, 'session'):
            session = requests.Session()
            # Increased pool size for single high-throughput server
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=50,  # More connections for single server
                pool_maxsize=100,     # Higher max pool
                max_retries=requests.adapters.Retry(
                    total=1,
                    backoff_factor=0.1,
                    status_forcelist=[500, 502, 503, 504]
                )
            )
            session.mount('http://', adapter)
            self._local.session = session
        return self._local.session

    def _flush_logs(self):
        """Force flush all log handlers"""
        for handler in logger.handlers:
            handler.flush()

    def get_usage_inst(self):
        return "You can search for information by putting your query between <search> and </search> tags."
    
    def _parse_search_query(self, action: str) -> Tuple[str, bool]:
        if "</search>" in action:
            search_matches = re.findall(r"<search>(.*?)</search>", action, re.DOTALL)
            if len(search_matches) > 0:
                return search_matches[-1].strip(), True
        return "", False
    
    def _parse_answer_tags(self, action: str) -> Tuple[str, bool]:
        if "</answer>" in action:
            answer_matches = re.findall(r"<answer>(.*?)</answer>", action, re.DOTALL)
            if len(answer_matches) > 0:
                return answer_matches[-1].strip(), True
        return "", False
    
    def parse_action(self, action: str) -> Tuple[str, bool]:
        query, valid = self._parse_search_query(action)
        if valid: return query, True
        return self._parse_answer_tags(action)
    
    def get_action_priority(self, action: str, extra_field: dict) -> int:
        if "</search>" in action:
            _, valid = self._parse_search_query(action)
            if valid: return 100
        _, valid = self.parse_action(action)
        return 0 if valid else -1

    def _get_cache_key(self, query: str) -> str:
        return hashlib.md5(query.encode('utf-8')).hexdigest()

    def conduct_action(self, trajectory_id, action, extra_field):
        parsed_query, is_valid = self._parse_search_query(action)
        env = self.load_env(trajectory_id)
        
        if not is_valid:
            parsed_query, is_valid = self._parse_answer_tags(action)
            if is_valid:
                logger.debug(f"[Trajectory {trajectory_id}] Answer tag detected")
                observation, done, valid = "", True, False 
            else:
                logger.warning(f"[Trajectory {trajectory_id}] Invalid action format")
                observation, done, valid = "", False, False
        else:
            try:
                # Thread-safe statistics increment
                with self.stats_lock:
                    self.total_searches += 1
                    current_total = self.total_searches
                
                logger.info(f"[Trajectory {trajectory_id}] Query: {parsed_query[:100]}")
                
                # --- CACHE CHECK (Thread-Safe) ---
                cache_key = self._get_cache_key(parsed_query)
                cached_result = None
                
                with self.cache_lock:
                    if cache_key in self.query_cache:
                        cached_result = self.query_cache[cache_key]
                        # Move to end to mark as recently used
                        self.query_cache.move_to_end(cache_key)
                
                if cached_result:
                    with self.stats_lock:
                        self.cache_hits += 1
                    search_results = [cached_result]
                    logger.debug(f"[Trajectory {trajectory_id}] Cache HIT")
                else:
                    with self.stats_lock:
                        self.cache_misses += 1
                    
                    search_results = self._batch_search([parsed_query])
                    
                    # Store in cache if valid result
                    if search_results and search_results[0]:
                        with self.cache_lock:
                            # FIFO eviction if cache full
                            if len(self.query_cache) >= self.max_cache_size:
                                self.query_cache.popitem(last=False)
                            self.query_cache[cache_key] = search_results[0]

                # --- RESULT PROCESSING ---
                if not search_results or not search_results[0]:
                    with self.stats_lock:
                        self.empty_searches += 1
                    self._log_empty_query(trajectory_id, parsed_query, extra_field)
                    formatted_results = ""
                    logger.warning(f"[Trajectory {trajectory_id}] EMPTY RESULT")
                else:
                    formatted_results = self._passages2string(search_results[0])
                
                observation = f'\n\n<information>{formatted_results.strip()}</information>\n\n'
                execution_result = formatted_results
                done = False
                valid = True

            except Exception as e:
                with self.stats_lock:
                    self.failed_searches += 1
                logger.error(f"[Trajectory {trajectory_id}] SEARCH FAILED: {e}", exc_info=True)
                self._flush_logs()
                
                execution_result = f"Search error: {str(e)}"
                observation = f'\n\n<information>Search temporarily unavailable</information>\n\n'
                done = False
                valid = False
        
        self.update_env(trajectory_id, env, parsed_query, is_valid, extra_field, execution_result)
        self.save_env(trajectory_id, env)
        
        # Log stats every 100 searches
        with self.stats_lock:
            if self.total_searches > 0 and self.total_searches % 100 == 0:
                should_log = True
            else:
                should_log = False
        
        if should_log:
            self._log_stats()
            
        return observation, done, valid

    def _log_stats(self):
        """Writes current statistics to the log file (thread-safe)"""
        with self.stats_lock:
            total_queries = self.cache_hits + self.cache_misses
            cache_rate = 100 * self.cache_hits / total_queries if total_queries > 0 else 0
            empty_rate = 100 * self.empty_searches / self.total_searches if self.total_searches > 0 else 0
            timeout_rate = 100 * self.timeout_count / self.cache_misses if self.cache_misses > 0 else 0
            
            stats_msg = (
                f"\n=== Search Statistics Update ===\n"
                f"Total Requests: {self.total_searches}\n"
                f"Cache Hits: {self.cache_hits} ({cache_rate:.1f}%)\n"
                f"Cache Misses: {self.cache_misses} (actual server calls)\n"
                f"Empty Results: {self.empty_searches} ({empty_rate:.1f}%)\n"
                f"Failed: {self.failed_searches}\n"
                f"Timeouts: {self.timeout_count} ({timeout_rate:.1f}% of server calls)\n"
                f"Retries: {self.retry_count}\n"
                f"Cache Size: {len(self.query_cache)}/{self.max_cache_size}\n"
                f"Cache Efficiency: {cache_rate:.1f}% (higher = less server load)\n"
                f"================================\n"
            )
        
        logger.info(stats_msg)
        self._flush_logs()

    def _log_empty_query(self, trajectory_id, query, extra_field):
        """Log empty query to a separate file for analysis (thread-safe)"""
        try:
            with self.empty_log_lock:
                with open(self.empty_queries_file, 'a') as f:
                    f.write(f"Timestamp: {datetime.now()} | Traj: {trajectory_id}\n")
                    f.write(f"Query: {query}\n")
                    f.write("-" * 40 + "\n")
                    f.flush()
        except Exception as e:
            logger.error(f"Failed to log empty query: {e}")

    def _batch_search(self, queries: List[str]) -> List[List[Dict]]:
        """Call retrieval service with retry logic (thread-safe, multi-GPU single server)"""
        url = self.retriever_url
        
        payload = {"queries": queries, "topk": self.topk, "return_scores": True}
        
        # Get thread-local session
        session = self._get_session()
        
        # Retry up to 3 times with exponential backoff
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Progressive timeout: multi-GPU server can handle parallel, but large index needs time
                # Start conservative, increase on retry
                timeout = 60 + (attempt * 30)  # 60s, 90s, 120s
                
                response = session.post(f"{url}/retrieve", json=payload, timeout=timeout)
                response.raise_for_status()
                
                # Success - return results
                return response.json().get('result', [[]])
                
            except requests.exceptions.Timeout:
                with self.stats_lock:
                    self.timeout_count += 1
                    if attempt < max_retries - 1:
                        self.retry_count += 1
                
                logger.warning(f"Timeout at {url} (attempt {attempt+1}/{max_retries}, timeout={timeout}s)")
                
                if attempt < max_retries - 1:
                    backoff = 0.5 * (2 ** attempt)  # Exponential: 0.5s, 1.0s, 2.0s
                    time.sleep(backoff)
                    
            except requests.exceptions.RequestException as e:
                with self.stats_lock:
                    if attempt < max_retries - 1:
                        self.retry_count += 1
                
                logger.warning(f"Request error at {url} (attempt {attempt+1}/{max_retries}): {e}")
                
                if attempt < max_retries - 1:
                    backoff = 0.5 * (2 ** attempt)
                    time.sleep(backoff)
                    
            except Exception as e:
                logger.error(f"Unexpected error at {url}: {e}")
                break  # Don't retry on unexpected errors
        
        # All retries exhausted
        logger.error(f"Retrieval server failed after {max_retries} attempts.")
        self._flush_logs()
        return [[] for _ in queries]

    def _passages2string(self, retrieval_result: List[Dict]) -> str:
        if not retrieval_result: return ''
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
            try:
                if 'document' in doc_item:
                    doc = doc_item['document']
                    doc_id = doc.get('id', 'Unknown')
                    content = doc.get('text', 'No content')
                else:
                    content = doc_item.get('text', str(doc_item))
                    doc_id = "Unknown"
                format_reference += f"Doc {idx+1} (ID: {doc_id}):\n{content}\n\n"
            except Exception:
                pass
        return format_reference
    
    def __del__(self):
        """Cleanup: close all thread-local sessions"""
        try:
            if hasattr(self._local, 'session'):
                self._local.session.close()
        except:
            pass
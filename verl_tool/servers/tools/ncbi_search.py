"""
Enhanced NCBI Search Tool with Better Error Handling and Debugging
"""
import os
import json
import time
import pathlib
import asyncio
import aiofiles
import aiohttp
from typing import Optional, Dict, List, Any, Tuple
import regex as re
import logging
from xml.etree import ElementTree as ET
from collections import OrderedDict

from .base import BaseTool, register_tool

# Configure logging with more detail
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AsyncLRUCache:
    """Thread-safe LRU cache for async operations"""
    
    def __init__(self, max_size: int = 10000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache = OrderedDict()
        self._timestamps = {}
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        async with self._lock:
            if key in self._cache:
                if time.time() - self._timestamps[key] > self.ttl_seconds:
                    del self._cache[key]
                    del self._timestamps[key]
                    return None
                self._cache.move_to_end(key)
                return self._cache[key]
            return None
    
    async def set(self, key: str, value: Any):
        async with self._lock:
            while len(self._cache) >= self.max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                del self._timestamps[oldest_key]
            self._cache[key] = value
            self._timestamps[key] = time.time()
    
    def __len__(self):
        return len(self._cache)


class NCBISearchEngine:
    """Enhanced NCBI E-utilities search engine with robust error handling"""
    
    def __init__(
        self,
        email: str,
        api_key: Optional[str] = None,
        database: str = "pubmed",
        max_results: int = 3,
        cache_file: Optional[str] = None,
        cache_size: int = 10000,
        cache_ttl: int = 3600,
        min_timeout: int = 10,
        max_timeout: int = 45,
        enable_fallback: bool = True
    ):
        """Initialize NCBI search engine with enhanced error handling."""
        self.email = email
        self.api_key = api_key
        self.database = database
        self.max_results = max_results
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.enable_fallback = enable_fallback
        
        # Rate limiting
        self.requests_per_second = 10 if api_key else 3
        self._last_request_time = 0
        self._request_lock = asyncio.Lock()
        self._request_times = []
        
        # Adaptive timeout
        self.min_timeout = min_timeout
        self.max_timeout = max_timeout
        
        # Enhanced circuit breaker with gradual recovery
        self.failure_count = 0
        self.failure_threshold = 10  # INCREASED from 5 to be more lenient
        self.circuit_open_time = None
        self.circuit_reset_timeout = 30  # REDUCED from 60 for faster recovery
        self.consecutive_successes = 0
        self.required_successes_to_close = 3  # Need 3 successes to fully recover
        
        # Caching
        self._memory_cache = AsyncLRUCache(cache_size, cache_ttl)
        self._setup_cache_file(cache_file)
        
        # Enhanced statistics
        self._search_count = 0
        self.success_count = 0
        self.timeout_count = 0
        self.cache_hit_count = 0
        self.network_error_count = 0
        self.rate_limit_count = 0
        self.last_error = None
        self.last_error_time = None
        
        logger.info(f"NCBISearchEngine initialized: email={email}, db={database}, "
                   f"rate_limit={self.requests_per_second} req/s, "
                   f"timeout_range=[{min_timeout}, {max_timeout}]s, "
                   f"fallback={'enabled' if enable_fallback else 'disabled'}")
    
    def _setup_cache_file(self, cache_file: Optional[str]) -> None:
        """Set up cache file path."""
        if cache_file is None:
            cache_dir = pathlib.Path.home() / ".verl_cache"
            cache_dir.mkdir(exist_ok=True)
            self._cache_file = cache_dir / f"ncbi_{self.database}_cache.jsonl"
        else:
            self._cache_file = pathlib.Path(cache_file)
            self._cache_file.parent.mkdir(parents=True, exist_ok=True)
    
    async def _load_persistent_cache(self) -> None:
        """Load cache from file asynchronously."""
        if not self._cache_file.exists():
            logger.info(f"No existing cache file at {self._cache_file}")
            return
        
        try:
            async with aiofiles.open(self._cache_file, "r", encoding="utf-8") as f:
                cache_entries = 0
                async for line in f:
                    if line.strip():
                        try:
                            item = json.loads(line)
                            await self._memory_cache.set(item['query'], item['result'])
                            cache_entries += 1
                        except json.JSONDecodeError:
                            continue
                
                logger.info(f"Loaded {cache_entries} NCBI cache entries from {self._cache_file}")
        except Exception as e:
            logger.error(f"Failed to load NCBI cache: {e}")
    
    async def _append_to_persistent_cache(self, query: str, result: str) -> None:
        """Append to persistent cache asynchronously."""
        try:
            entry = {"query": query, "result": result, "timestamp": time.time()}
            async with aiofiles.open(self._cache_file, "a", encoding="utf-8") as f:
                await f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"NCBI cache write failed: {e}")
    
    async def _rate_limit(self):
        """Enforce NCBI rate limits with monitoring."""
        async with self._request_lock:
            now = time.time()
            self._request_times = [t for t in self._request_times if now - t < 1.0]
            
            current_rate = len(self._request_times)
            if current_rate >= self.requests_per_second:
                wait_time = 1.0 - (now - self._request_times[0])
                if wait_time > 0:
                    logger.debug(f"Rate limit: waiting {wait_time:.3f}s (current: {current_rate}/{self.requests_per_second} req/s)")
                    await asyncio.sleep(wait_time)
            
            self._request_times.append(time.time())
            self._last_request_time = time.time()
    
    def _check_circuit_breaker(self) -> bool:
        """Check if circuit breaker is open. Returns True if open."""
        if self.circuit_open_time:
            elapsed = time.time() - self.circuit_open_time
            if elapsed < self.circuit_reset_timeout:
                logger.warning(f"Circuit breaker OPEN - {self.circuit_reset_timeout - elapsed:.1f}s remaining")
                return True
            else:
                # Try to reset
                logger.info(f"Circuit breaker attempting reset (had {self.failure_count} failures)")
                self.circuit_open_time = None
                self.failure_count = max(0, self.failure_count // 2)  # Gradual reduction
                self.consecutive_successes = 0
                return False
        return False
    
    def _record_success(self):
        """Record a successful request."""
        self.success_count += 1
        self.consecutive_successes += 1
        
        # Gradually reduce failure count on success
        if self.consecutive_successes >= self.required_successes_to_close:
            self.failure_count = max(0, self.failure_count - 1)
            self.consecutive_successes = 0
            if self.failure_count == 0:
                logger.info("Circuit breaker fully recovered")
    
    def _record_failure(self, error_type: str, error_msg: str):
        """Record a failed request."""
        self.failure_count += 1
        self.consecutive_successes = 0
        self.last_error = f"{error_type}: {error_msg}"
        self.last_error_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            if not self.circuit_open_time:
                self.circuit_open_time = time.time()
                logger.error(f"Circuit breaker OPENED after {self.failure_count} failures. Last error: {self.last_error}")
    
    async def _make_request(self, endpoint: str, params: Dict, timeout: int = 45) -> str:
        """Make HTTP request with enhanced error handling and logging."""
        
        # Check circuit breaker
        if self._check_circuit_breaker():
            raise Exception(f"Circuit breaker open. Last error: {self.last_error}")
        
        start_time = time.time()
        params['email'] = self.email
        if self.api_key:
            params['api_key'] = self.api_key
        
        url = f"{self.base_url}{endpoint}"
        max_retries = 5  # INCREASED from 3
        base_delay = 1.0  # INCREASED back to 1.0 for better spacing
        
        last_exception = None
        
        for attempt in range(max_retries + 1):
            elapsed = time.time() - start_time
            remaining = timeout - elapsed
            
            if remaining <= 1.0:
                self._record_failure("timeout", f"Total budget {timeout}s exceeded")
                raise asyncio.TimeoutError(f"Total budget of {timeout}s exceeded after {attempt} attempts")

            timeout_config = aiohttp.ClientTimeout(
                total=remaining,
                connect=10,  # 10s connection timeout
                sock_read=30  # 30s read timeout
            )
            
            await self._rate_limit()

            try:
                logger.debug(f"NCBI request attempt {attempt+1}/{max_retries+1}: {endpoint} with params: {params}")
                
                async with aiohttp.ClientSession(timeout=timeout_config) as session:
                    async with session.get(url, params=params) as response:
                        logger.debug(f"NCBI response status: {response.status}")
                        
                        if response.status == 200:
                            text = await response.text()
                            logger.debug(f"NCBI response length: {len(text)} chars")
                            self._record_success()
                            return text
                        
                        elif response.status == 429:
                            self.rate_limit_count += 1
                            wait_time = base_delay * (2 ** attempt)
                            import random
                            wait_time += random.uniform(0, 1.0)
                            
                            logger.warning(f"NCBI Rate Limit (429). Waiting {wait_time:.2f}s... (attempt {attempt+1}/{max_retries+1})")
                            await asyncio.sleep(wait_time)
                            continue
                        
                        elif response.status in [502, 503, 504]:
                            # Server error - retry with backoff
                            wait_time = base_delay * (2 ** attempt)
                            logger.warning(f"NCBI server error {response.status}. Waiting {wait_time:.2f}s...")
                            await asyncio.sleep(wait_time)
                            continue
                        
                        else:
                            text = await response.text()
                            error_msg = f"HTTP {response.status}: {text[:200]}"
                            logger.error(error_msg)
                            self._record_failure(f"HTTP_{response.status}", text[:100])
                            raise Exception(error_msg)

            except asyncio.TimeoutError as e:
                last_exception = e
                self.timeout_count += 1
                logger.warning(f"Timeout on attempt {attempt+1}/{max_retries+1}: {e}")
                if attempt < max_retries:
                    await asyncio.sleep(1.0)
                continue
                
            except aiohttp.ClientError as e:
                last_exception = e
                self.network_error_count += 1
                logger.warning(f"Network error on attempt {attempt+1}/{max_retries+1}: {e}")
                if attempt < max_retries:
                    await asyncio.sleep(1.0)
                continue
            
            except Exception as e:
                last_exception = e
                logger.error(f"Unexpected error on attempt {attempt+1}/{max_retries+1}: {e}")
                if attempt < max_retries:
                    await asyncio.sleep(1.0)
                continue
        
        # All retries exhausted
        error_msg = f"Max retries ({max_retries}) exceeded. Last error: {last_exception}"
        logger.error(error_msg)
        self._record_failure("max_retries", str(last_exception))
        raise Exception(error_msg)
    
    def _create_fallback_result(self, query: str) -> str:
        """Create a fallback result when search fails."""
        return (f"Unable to retrieve PubMed results for query: '{query}'. "
                f"The search service is temporarily unavailable. "
                f"This is a placeholder response to allow training to continue.")
    
    async def execute(self, query: str, timeout: int = 30) -> str:
        """Execute NCBI search with enhanced error handling."""
        query = query.strip()
        if not query:
            return "Empty search query provided."
        
        if len(query) > 500:
            return "Search query too long (maximum 500 characters)."
        
        # Clamp timeout
        timeout = max(self.min_timeout, min(timeout, self.max_timeout))
        
        try:
            # Check cache first
            cached_result = await self._memory_cache.get(query)
            if cached_result is not None:
                logger.debug(f"NCBI cache hit for: {query}")
                self.cache_hit_count += 1
                return cached_result
            
            # Make API requests
            logger.info(f"NCBI API call for: {query} (timeout={timeout}s)")
            articles = await self._search_pubmed(query, timeout)
            
            # Format results
            result = self._format_articles(articles)
            
            # Cache results
            await self._cache_results(query, result)
            
            return result
        
        except asyncio.TimeoutError as e:
            logger.warning(f"NCBI search timeout for: {query}")
            if self.enable_fallback:
                return self._create_fallback_result(query)
            return "Search request timed out. Try a more specific query."
        
        except Exception as e:
            logger.error(f"NCBI search failed for '{query}': {e}")
            if self.enable_fallback:
                return self._create_fallback_result(query)
            return f"Search temporarily unavailable: {str(e)[:100]}"
    
    async def _search_pubmed(self, query: str, timeout: int) -> List[Dict]:
        """Search PubMed via NCBI E-utilities."""
        # Split timeout between search and fetch
        search_timeout = timeout // 2
        fetch_timeout = timeout - search_timeout
        
        # Step 1: Search for PMIDs
        pmids = await self._esearch(query, search_timeout)
        
        if not pmids:
            logger.warning(f"No NCBI results found for: {query}")
            return []
        
        # Step 2: Fetch article details
        articles = await self._efetch(pmids, fetch_timeout)
        
        return articles
    
    async def _esearch(self, query: str, timeout: int) -> List[str]:
        """Search and return PMIDs."""
        params = {
            "db": self.database,
            "term": query,
            "retmax": self.max_results,
            "retmode": "json"
        }
        
        response_text = await self._make_request("esearch.fcgi", params, timeout)
        data = json.loads(response_text)
        pmids = data.get("esearchresult", {}).get("idlist", [])
        
        logger.debug(f"Found {len(pmids)} PMIDs for: {query}")
        return pmids
    
    async def _efetch(self, pmids: List[str], timeout: int) -> List[Dict]:
        """Fetch article details."""
        if not pmids:
            return []
        
        params = {
            "db": self.database,
            "id": ",".join(pmids),
            "retmode": "xml"
        }
        
        response_text = await self._make_request("efetch.fcgi", params, timeout)
        return self._parse_pubmed_xml(response_text)
    
    def _parse_pubmed_xml(self, xml_text: str) -> List[Dict]:
        """Parse PubMed XML response."""
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            logger.error(f"XML parse error: {e}")
            return []
        
        articles = []
        for article_elem in root.findall(".//PubmedArticle"):
            try:
                pmid_elem = article_elem.find(".//PMID")
                pmid = pmid_elem.text if pmid_elem is not None else "Unknown"
                
                title_elem = article_elem.find(".//ArticleTitle")
                title = title_elem.text if title_elem is not None else "No title"
                
                abstract_texts = article_elem.findall(".//AbstractText")
                if abstract_texts:
                    abstract = " ".join([elem.text for elem in abstract_texts if elem.text])
                else:
                    abstract = "No abstract available"
                
                pub_date = article_elem.find(".//PubDate")
                year = "Unknown"
                if pub_date is not None:
                    year_elem = pub_date.find("Year")
                    if year_elem is not None:
                        year = year_elem.text
                
                articles.append({
                    "pmid": pmid,
                    "title": title,
                    "abstract": abstract,
                    "year": year
                })
            except Exception as e:
                logger.error(f"Error parsing article: {e}")
                continue
        
        return articles
    
    def _format_articles(self, articles: List[Dict]) -> str:
        """Format articles for model consumption."""
        if not articles:
            return "No articles found in PubMed for this query."
        
        formatted_parts = []
        for idx, article in enumerate(articles, 1):
            title = article.get('title', 'No title')
            abstract = article.get('abstract', 'No abstract')
            pmid = article.get('pmid', 'Unknown')
            year = article.get('year', 'Unknown')
            
            if len(abstract) > 800:
                abstract = abstract[:800] + "..."
            
            formatted_parts.append(
                f"Doc {idx}(Title: {title}, PMID: {pmid}, Year: {year}) {abstract}"
            )
        
        return "\n\n".join(formatted_parts)
    
    async def _cache_results(self, query: str, result: str) -> None:
        """Cache results in memory and persistent storage."""
        try:
            await self._memory_cache.set(query, result)
            await self._append_to_persistent_cache(query, result)
            self._search_count += 1
        except Exception as e:
            logger.error(f"NCBI caching failed: {e}")
    
    def get_stats(self) -> dict:
        """Get comprehensive performance statistics."""
        total = self.success_count + self.timeout_count
        cache_hit_rate = self.cache_hit_count / max(1, self.cache_hit_count + total)
        
        stats = {
            'total_searches': self._search_count,
            'success_count': self.success_count,
            'timeout_count': self.timeout_count,
            'network_error_count': self.network_error_count,
            'rate_limit_count': self.rate_limit_count,
            'cache_hit_count': self.cache_hit_count,
            'cache_hit_rate': f"{cache_hit_rate:.2%}",
            'cache_size': len(self._memory_cache),
            'circuit_breaker_open': self.circuit_open_time is not None,
            'failure_count': self.failure_count,
            'consecutive_successes': self.consecutive_successes,
            'last_error': self.last_error,
            'last_error_time': self.last_error_time
        }
        
        return stats


@register_tool
class NCBISearchTool(BaseTool):
    """NCBI PubMed search tool with enhanced debugging."""
    
    tool_type = "ncbi_search"
    
    def __init__(
        self,
        num_workers=1,
        ncbi_email: str = None,
        ncbi_api_key: str = None,
        database: str = "pubmed",
        max_results: int = 3,
        cache_file: Optional[str] = None,
        default_timeout: int = 30,
        cache_size: int = 10000,
        cache_ttl: int = 3600,
        min_timeout: int = 10,
        max_timeout: int = 45,
        enable_fallback: bool = True
    ):
        """Initialize NCBI search tool."""
        super().__init__(num_workers)
        
        if ncbi_email is None:
            ncbi_email = os.getenv('NCBI_EMAIL')
            if ncbi_email is None:
                raise ValueError(
                    "NCBI email required: set NCBI_EMAIL environment variable"
                )
        
        self.search_engine = NCBISearchEngine(
            email=ncbi_email,
            api_key=ncbi_api_key or os.getenv('NCBI_API_KEY'),
            database=database,
            max_results=max_results,
            cache_file=cache_file,
            cache_size=cache_size,
            cache_ttl=cache_ttl,
            min_timeout=min_timeout,
            max_timeout=max_timeout,
            enable_fallback=enable_fallback
        )
        
        self.default_timeout = default_timeout
        self._initialized = False
        self._init_lock = asyncio.Lock()
        
        requests_per_second = 10 if self.search_engine.api_key else 3
        self.semaphore = asyncio.Semaphore(requests_per_second * 2)
        
        logger.info(f"NCBISearchTool initialized with {num_workers} workers, "
                   f"default_timeout={default_timeout}s, fallback={'enabled' if enable_fallback else 'disabled'}")
    
    async def _ensure_initialized(self):
        """Lazy initialization."""
        if not self._initialized:
            async with self._init_lock:
                if not self._initialized:
                    await self.search_engine._load_persistent_cache()
                    self._initialized = True
                    logger.info("NCBI search engine initialized and cache loaded")
    
    def get_usage_inst(self):
        return "Search PubMed medical literature using <search>your query</search> format."
    
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
        search_query, is_valid = self._parse_search_query(action)
        if is_valid:
            return search_query, True
        answer, is_valid = self._parse_answer_tags(action)
        if is_valid:
            return answer, True
        return "", False
    
    def get_action_priority(self, action: str, extra_field: dict) -> int:
        if "</search>" in action:
            _, valid = self.parse_action(action)
            if valid:
                return 100
        _, valid = self.parse_action(action)
        return 0 if valid else -1
    
    async def aget_observations(
        self,
        trajectory_ids: List[str],
        actions: List[str],
        extra_fields: List[Dict[str, Any]]
    ) -> Tuple[List[str], List[bool], List[bool]]:
        """Process multiple search actions concurrently."""
        await self._ensure_initialized()
        
        async def process_single_action(trajectory_id, action, extra_field):
            async with self.semaphore:
                try:
                    return await self._conduct_action_async(trajectory_id, action, extra_field)
                except Exception as e:
                    logger.error(f"NCBI search error for traj {trajectory_id}: {e}", exc_info=True)
                    return f"Search error: {str(e)}", False, False
        
        tasks = [
            process_single_action(trajectory_id, action, extra_field)
            for trajectory_id, action, extra_field in zip(trajectory_ids, actions, extra_fields)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        observations, dones, valids = [], [], []
        for result in results:
            if isinstance(result, Exception):
                obs, done, valid = f"Search error: {str(result)}", False, False
            else:
                obs, done, valid = result
            
            observations.append(obs)
            dones.append(done)
            valids.append(valid)
        
        # Log stats every 50 searches
        if len(trajectory_ids) > 0 and self.search_engine._search_count % 50 == 0:
            stats = self.search_engine.get_stats()
            logger.info(f"NCBI Stats: {stats}")
        
        self.maybe_cleanup_env(trajectory_ids, actions, extra_fields)
        
        return observations, dones, valids
    
    async def _conduct_action_async(
        self,
        trajectory_id: str,
        action: str,
        extra_field: Dict[str, Any]
    ) -> Tuple[str, bool, bool]:
        """Conduct single search action asynchronously."""
        parsed_query, is_valid = self._parse_search_query(action)
        env = self.load_env(trajectory_id)
        
        if not is_valid:
            parsed_query, is_valid = self._parse_answer_tags(action)
            if is_valid:
                return "", True, False
            else:
                return "", False, False
        
        timeout = extra_field.get('timeout', self.default_timeout)
        timeout = max(self.search_engine.min_timeout, 
                     min(timeout, self.search_engine.max_timeout))
        
        try:
            search_results = await self.search_engine.execute(parsed_query, timeout)
            observation = f'\n\n<information>{search_results.strip()}</information>\n\n'
            done, valid = False, True
        except Exception as e:
            logger.error(f"NCBI execution error: {e}", exc_info=True)
            if self.search_engine.enable_fallback:
                fallback = self.search_engine._create_fallback_result(parsed_query)
                observation = f'\n\n<information>{fallback}</information>\n\n'
            else:
                observation = f'\n\n<information>Search temporarily unavailable.</information>\n\n'
            done, valid = False, False
        
        self.update_env(trajectory_id, env, action, is_valid, extra_field, observation)
        self.save_env(trajectory_id, env)
        
        return observation, done, valid
    
    def conduct_action(
        self,
        trajectory_id: str,
        action: str,
        extra_field: Dict[str, Any]
    ) -> Tuple[str, bool, bool]:
        """Synchronous wrapper for async operations."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import threading
                
                result = [None]
                exception = [None]
                
                def run_in_new_loop():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        result[0] = new_loop.run_until_complete(
                            self._conduct_action_async(trajectory_id, action, extra_field)
                        )
                    except Exception as e:
                        exception[0] = e
                    finally:
                        new_loop.close()
                
                thread = threading.Thread(target=run_in_new_loop)
                thread.start()
                thread.join(timeout=self.search_engine.max_timeout + 5)
                
                if exception[0]:
                    raise exception[0]
                if result[0] is None:
                    return "NCBI search timed out", False, False
                return result[0]
            else:
                return loop.run_until_complete(
                    self._conduct_action_async(trajectory_id, action, extra_field)
                )
        except RuntimeError:
            return asyncio.run(self._conduct_action_async(trajectory_id, action, extra_field))
        except Exception as e:
            logger.error(f"NCBI search failed: {e}", exc_info=True)
            return f"NCBI search failed: {str(e)}", False, False
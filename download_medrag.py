#!/usr/bin/env python3
"""
Download MedRAG dataset and prepare for indexing
MedRAG provides pre-chunked medical documents perfect for retrieval
"""

import json
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
import argparse

def download_and_prepare_medrag(
    output_dir: str = "./data/search_r1/retriever_index",
    corpus_name: str = "pubmed",
    max_chunks: int = None
):
    """
    Download MedRAG dataset and convert to the format needed for indexing.
    
    Args:
        output_dir: Where to save the corpus file
        corpus_name: Which corpus to use ('pubmed', 'textbooks')
        max_chunks: Maximum number of chunks to process (None = all)
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print(f"Downloading MedRAG {corpus_name.title()} Data")
    print("="*70)
    print(f"Corpus: {corpus_name}")
    print(f"Output directory: {output_dir}")
    print(f"Max chunks: {max_chunks if max_chunks else 'all'}")
    print("="*70)
    print()
    
    # Download MedRAG dataset
    print(f"Downloading MedRAG/{corpus_name} from HuggingFace...")
    print("This may take a while depending on your connection...")
    print()
    
    try:
        dataset = load_dataset(f"MedRAG/{corpus_name}", split="train")
        print(f"✓ Downloaded {len(dataset)} chunks")
        print()
        
    except Exception as e:
        print(f"ERROR downloading dataset: {e}")
        print()
        print("Make sure you have:")
        print("  pip install datasets")
        print()
        print("Available MedRAG corpora:")
        print("  - MedRAG/pubmed (PubMed abstracts)")
        print("  - MedRAG/textbooks (medical textbooks)")
        return
    
    # Convert to corpus format
    corpus_file = output_dir / f"{corpus_name}.jsonl"
    
    print(f"Converting to corpus format...")
    print(f"Output: {corpus_file}")
    print()
    
    with open(corpus_file, 'w', encoding='utf-8') as f:
        for idx, item in enumerate(tqdm(dataset, desc=f"Processing {corpus_name}")):
            if max_chunks and idx >= max_chunks:
                break
            
            # MedRAG format: 'id', 'title', 'content' or 'text'
            doc_id = item.get('id', f"{corpus_name}_{idx}")
            title = item.get('title', '')
            content = item.get('content', item.get('text', ''))
            
            # Create combined text
            if title and title not in content:
                text = f"{title}. {content}"
            else:
                text = content
            
            # Write in format expected by indexer
            doc = {
                'id': doc_id,
                'text': text
            }
            
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')
    
    print(f"✓ Saved {idx + 1} documents to {corpus_file}")
    print()
    
    # Show statistics
    file_size_mb = corpus_file.stat().st_size / (1024 * 1024)
    print("="*70)
    print("Statistics:")
    print(f"  Total documents: {idx + 1}")
    print(f"  File size: {file_size_mb:.2f} MB")
    print(f"  Output file: {corpus_file}")
    print("="*70)
    print()
    
    # Show sample
    print("Sample documents:")
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            doc = json.loads(line)
            print(f"\n{i+1}. ID: {doc['id']}")
            print(f"   Text: {doc['text'][:150]}...")
    print()
    
    return corpus_file


def download_combined_corpus(output_dir: str = "./data/search_r1/retriever_index"):
    """
    Download textbooks and PubMed, combine them into one corpus.
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    corpora = ['textbooks', 'pubmed']  # Removed 'wikipedia'
    combined_file = output_dir / "medical_combined.jsonl"
    
    print("="*70)
    print("Downloading Medical Corpora (Textbooks + PubMed)")
    print("="*70)
    print("This will download:")
    print("  - Medical textbooks (~200K chunks)")
    print("  - PubMed abstracts (~15M chunks)")
    print()
    print("Total size: ~40-80GB")
    print("This will take 1-3 hours depending on connection")
    print("="*70)
    print()
    
    response = input("Continue? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    doc_count = 0
    
    with open(combined_file, 'w', encoding='utf-8') as out_f:
        for corpus_name in corpora:
            print(f"\n{'='*70}")
            print(f"Processing: {corpus_name}")
            print('='*70)
            
            try:
                dataset = load_dataset(f"MedRAG/{corpus_name}", split="train")
                print(f"✓ Loaded {len(dataset)} chunks from {corpus_name}")
                
                for idx, item in enumerate(tqdm(dataset, desc=f"Processing {corpus_name}")):
                    doc_id = item.get('id', f"{corpus_name}_{idx}")
                    title = item.get('title', '')
                    content = item.get('content', item.get('text', ''))
                    
                    if title and title not in content:
                        text = f"{title}. {content}"
                    else:
                        text = content
                    
                    doc = {
                        'id': doc_id,
                        'text': text,
                        'source': corpus_name
                    }
                    
                    out_f.write(json.dumps(doc, ensure_ascii=False) + '\n')
                    doc_count += 1
                
            except Exception as e:
                print(f"ERROR with {corpus_name}: {e}")
                print("Continuing with next corpus...")
                continue
    
    file_size_gb = combined_file.stat().st_size / (1024 * 1024 * 1024)
    
    print()
    print("="*70)
    print("✓ Combined corpus created!")
    print("="*70)
    print(f"  Total documents: {doc_count:,}")
    print(f"  File size: {file_size_gb:.2f} GB")
    print(f"  Output: {combined_file}")
    print("="*70)
    print()
    
    return combined_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download MedRAG data")
    parser.add_argument(
        "--output_dir", 
        default="./data/search_r1/retriever_index",
        help="Output directory for corpus"
    )
    parser.add_argument(
        "--corpus",
        default="pubmed",
        choices=['pubmed', 'textbooks', 'combined'],  # Changed 'all' to 'combined', removed 'wikipedia'
        help="Which corpus to download"
    )
    parser.add_argument(
        "--max_chunks",
        type=int,
        default=None,
        help="Maximum chunks to download (for testing)"
    )
    
    args = parser.parse_args()
    
    if args.corpus == 'combined':
        corpus_file = download_combined_corpus(args.output_dir)
    else:
        corpus_file = download_and_prepare_medrag(
            output_dir=args.output_dir,
            corpus_name=args.corpus,
            max_chunks=args.max_chunks
        )
    
    if corpus_file:
        print()
        print("Next steps:")
        print(f"1. Build index:")
        print(f"   python build_index.py \\")
        print(f"     --corpus_path {corpus_file} \\")
        print(f"     --index_output {Path(args.output_dir) / 'e5_Flat.index'} \\")
        print(f"     --model intfloat/e5-base-v2")
        print()
        print(f"2. Use in training:")
        print(f"   index_file={Path(args.output_dir) / 'e5_Flat.index'}")
        print(f"   corpus_file={corpus_file}")
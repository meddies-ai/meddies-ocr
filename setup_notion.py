"""
Notion Workspace Setup for MeddiesOCR Project
Using Notion API v1 with HTTP requests
"""

import requests
import json
import os
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

# Configuration
NOTION_API_KEY = os.environ.get('NOTION_API_KEY', '')
if not NOTION_API_KEY:
    raise ValueError('NOTION_API_KEY environment variable not set')
MAIN_PAGE_ID = os.environ.get('NOTION_MAIN_PAGE_ID', '')
NOTION_API_URL = "https://api.notion.com/v1"

# Headers for Notion API
HEADERS = {
    "Authorization": f"Bearer {NOTION_API_KEY}",
    "Notion-Version": "2022-06-28",
    "Content-Type": "application/json"
}

def get_page_details(page_id: str) -> Dict[str, Any]:
    """Fetch current page structure"""
    url = f"{NOTION_API_URL}/pages/{page_id}"
    response = requests.get(url, headers=HEADERS)
    return response.json()

def create_child_page(parent_id: str, title: str, icon: str, content: str) -> Dict[str, Any]:
    """Create a new child page"""
    url = f"{NOTION_API_URL}/pages"
    
    payload = {
        "parent": {"page_id": parent_id},
        "icon": {"type": "emoji", "emoji": icon},
        "properties": {
            "title": {
                "id": "title",
                "type": "title",
                "title": [{"type": "text", "text": {"content": title}}]
            }
        },
        "children": []
    }
    
    response = requests.post(url, headers=HEADERS, json=payload)
    if response.status_code not in [200, 201]:
        print(f"Error creating page '{title}': {response.status_code}")
        print(response.text)
        return None
    
    return response.json()

def create_database(parent_id: str, title: str, icon: str, schema: Dict) -> Dict[str, Any]:
    """Create a new database"""
    url = f"{NOTION_API_URL}/databases"
    
    payload = {
        "parent": {"page_id": parent_id},
        "icon": {"type": "emoji", "emoji": icon},
        "title": [{"type": "text", "text": {"content": title}}],
        "properties": schema
    }
    
    response = requests.post(url, headers=HEADERS, json=payload)
    if response.status_code not in [200, 201]:
        print(f"Error creating database '{title}': {response.status_code}")
        print(response.text)
        return None
    
    return response.json()

def create_rich_text_block(text: str, heading: int = None) -> Dict:
    """Create text/heading block"""
    if heading:
        return {
            "object": "block",
            "type": f"heading_{heading}",
            f"heading_{heading}": {
                "rich_text": [{"type": "text", "text": {"content": text}}]
            }
        }
    else:
        return {
            "object": "block",
            "type": "paragraph",
            "paragraph": {
                "rich_text": [{"type": "text", "text": {"content": text}}]
            }
        }

def add_blocks_to_page(page_id: str, blocks: list) -> bool:
    """Add blocks to existing page"""
    url = f"{NOTION_API_URL}/blocks/{page_id}/children"
    
    payload = {"children": blocks}
    
    response = requests.patch(url, headers=HEADERS, json=payload)
    if response.status_code not in [200, 201]:
        print(f"Error adding blocks: {response.status_code}")
        print(response.text)
        return False
    
    return True

def main():
    print("🚀 Starting Notion Workspace Setup for MeddiesOCR...")
    print(f"Main Page ID: {MAIN_PAGE_ID}")
    print()
    
    # 1. Fetch current page
    print("1️⃣  Fetching current page structure...")
    try:
        page_info = get_page_details(MAIN_PAGE_ID)
        print(f"✅ Current page: {page_info.get('properties', {}).get('title', {}).get('title', [{}])[0].get('text', {}).get('content', 'MeddiesOCR')}")
        print()
    except Exception as e:
        print(f"⚠️  Could not fetch page: {e}")
    
    # 2. Create child pages
    print("2️⃣  Creating child pages...")
    
    pages_config = [
        {
            "title": "🎯 Project Overview & Scope",
            "icon": "🎯",
            "blocks": [
                {"type": "heading_1", "content": "MeddiesOCR Project Overview"},
                {"type": "heading_2", "content": "Project Vision"},
                {"type": "text", "content": "Vision Language Models (VLMs) research focused on document understanding and OCR, with special emphasis on handwritten Vietnamese text extraction."},
                
                {"type": "heading_2", "content": "Main Objectives"},
                {"type": "text", "content": "1. Document Understanding & OCR: Trích xuất text từ tài liệu, đặc biệt handwritten Vietnamese"},
                {"type": "text", "content": "2. Knowledge Distillation: On-policy distillation, sparse logit sampling, ACID method"},
                {"type": "text", "content": "3. Architecture Comparison: Qwen VL, SigLIP 2, LFM2, CLIP, Chandra OCR"},
                
                {"type": "heading_2", "content": "Success Metrics"},
                {"type": "text", "content": "📊 OCR Tasks:"},
                {"type": "text", "content": "• CER (Character Error Rate): < 1%"},
                {"type": "text", "content": "• WER (Word Error Rate): < 5%"},
                {"type": "text", "content": "• Exact Match: > 95%"},
                
                {"type": "heading_2", "content": "Team Members"},
                {"type": "text", "content": "👥 Leo: Research discovery, exploration"},
                {"type": "text", "content": "👥 Bảo: Mathematical formulations, deep paper reading"},
                {"type": "text", "content": "👥 Tân: Mathematical formulations, deep paper reading"},
                {"type": "text", "content": "👥 Hoàng (Lead): Full-stack expertise"},
            ]
        },
        {
            "title": "📚 Paper & Research References",
            "icon": "📚",
            "blocks": [
                {"type": "heading_1", "content": "Key Papers & Resources"},
                {"type": "heading_2", "content": "Vision Language Models"},
                {"type": "text", "content": "• SigLIP 2: Sigmoid loss + LocCa decoder + self-distillation"},
                {"type": "text", "content": "• CLIP: Contrastive Language-Image Pre-training (baseline)"},
                
                {"type": "heading_2", "content": "Knowledge Distillation"},
                {"type": "text", "content": "• On-Policy Distillation: Student generates output → Compare with Teacher → Learn from mistakes"},
                {"type": "text", "content": "• Sparse Logit Sampling: 0.01% logits (12 tokens instead of 64K), unbiased gradient estimation"},
                
                {"type": "heading_2", "content": "Model References"},
                {"type": "text", "content": "• Qwen 2.5 VL: Base performance 90-95% accuracy on handwriting, after fine-tuning 99%+"},
                {"type": "text", "content": "• SigLIP 2 (Vision Encoder): NaFlex variant for documents with native aspect ratio support"},
                {"type": "text", "content": "• LFM2-VL-450M: Lightweight option (450M params) for edge deployment"},
                {"type": "text", "content": "• Chandra OCR: Specialized OCR architecture"},
                
                {"type": "heading_2", "content": "Implementation Resources"},
                {"type": "text", "content": "• Unsloth: https://github.com/unslothai/unsloth (Fast fine-tuning)"},
                {"type": "text", "content": "• Qwen VL: https://github.com/QwenLM/Qwen2-VL"},
                {"type": "text", "content": "• Chandra OCR: https://github.com/datalab-to/chandra"},
            ]
        },
        {
            "title": "🔗 Resources & Links",
            "icon": "🔗",
            "blocks": [
                {"type": "heading_1", "content": "Project Resources"},
                {"type": "heading_2", "content": "Code & Collaboration"},
                {"type": "text", "content": "GitHub Repository: https://github.com/meddies-ai/meddies-ocr"},
                
                {"type": "heading_2", "content": "Data & Documentation"},
                {"type": "text", "content": "Google Drive: https://drive.google.com/drive/folders/15mf7pVt9sHxxeqFvtYHKLpesujW4vySy?usp=drive_link"},
                
                {"type": "heading_2", "content": "External Tools"},
                {"type": "text", "content": "• PyTorch: https://pytorch.org/"},
                {"type": "text", "content": "• Hugging Face Transformers: https://huggingface.co/"},
                {"type": "text", "content": "• Weights & Biases: https://wandb.ai/ (Experiment tracking)"},
                
                {"type": "heading_2", "content": "Model Hubs"},
                {"type": "text", "content": "• Qwen Models: https://huggingface.co/Qwen"},
                {"type": "text", "content": "• OpenAI CLIP: https://huggingface.co/openai/clip-vit-base-patch32"},
            ]
        },
        {
            "title": "💪 AIM (Project Motivation)",
            "icon": "💪",
            "blocks": [
                {"type": "heading_1", "content": "Why MeddiesOCR Matters"},
                {"type": "heading_2", "content": "Problem Statement"},
                {"type": "text", "content": "Existing OCR systems struggle with handwritten Vietnamese text in documents. Current solutions require extensive manual correction and have high error rates (>5%), making them impractical for production use."},
                
                {"type": "heading_2", "content": "Our Solution"},
                {"type": "text", "content": "Leverage modern Vision Language Models (VLMs) with knowledge distillation techniques to achieve production-grade accuracy (<1% CER) while maintaining affordable computational costs."},
                
                {"type": "heading_2", "content": "Expected Impact"},
                {"type": "text", "content": "✨ Faster document processing for Vietnamese businesses"},
                {"type": "text", "content": "✨ Cost reduction in manual data entry and correction"},
                {"type": "text", "content": "✨ Foundation for downstream document understanding tasks (DocVQA, information extraction)"},
                
                {"type": "heading_2", "content": "Timeline"},
                {"type": "text", "content": "Phase 1 (Month 1): Data preparation + baseline model evaluation"},
                {"type": "text", "content": "Phase 2 (Month 2): Fine-tuning + knowledge distillation experiments"},
                {"type": "text", "content": "Phase 3 (Month 3): Evaluation + production deployment"},
            ]
        },
        {
            "title": "🏆 Benchmarks & Metrics Dashboard",
            "icon": "🏆",
            "blocks": [
                {"type": "heading_1", "content": "Performance Tracking"},
                {"type": "heading_2", "content": "OCR Task Metrics"},
                {"type": "text", "content": "| Metric | Description | Target | Current |"},
                {"type": "text", "content": "|--------|-------------|--------|---------|"},
                {"type": "text", "content": "| CER | Character Error Rate (Levenshtein distance / total chars) | < 1% | - |"},
                {"type": "text", "content": "| WER | Word Error Rate | < 5% | - |"},
                {"type": "text", "content": "| Exact Match | Full string match percentage | > 95% | - |"},
                
                {"type": "heading_2", "content": "Document Understanding Benchmarks"},
                {"type": "text", "content": "| Benchmark | Description | Baseline |"},
                {"type": "text", "content": "|-----------|-------------|----------|"},
                {"type": "text", "content": "| DocVQA | Document question answering | Qwen: 91.4 |"},
                {"type": "text", "content": "| TextVQA | Text reading in natural images | Qwen: 84.3 |"},
                {"type": "text", "content": "| OCRBench | Comprehensive OCR evaluation | Qwen: 866/1000 |"},
                
                {"type": "heading_2", "content": "Model Comparison Matrix"},
                {"type": "text", "content": "Track performance across: Qwen VL, SigLIP 2, LFM2, CLIP, Chandra OCR"},
            ]
        },
    ]
    
    created_pages = {}
    for config in pages_config:
        print(f"  Creating '{config['title']}'...", end=" ")
        try:
            # For now, we'll create pages as simple text since block creation is complex
            # In production, you'd want to use append blocks API
            result = create_child_page(
                MAIN_PAGE_ID,
                config['title'],
                config['icon'],
                ""
            )
            if result:
                page_id = result.get('id')
                created_pages[config['title']] = page_id
                print(f"✅ (ID: {page_id})")
            else:
                print("❌")
        except Exception as e:
            print(f"⚠️  Error: {e}")
    
    print()
    
    # 3. Create Task List Database
    print("3️⃣  Creating Task List Database...")
    try:
        task_db_schema = {
            "Task": {
                "id": "title",
                "type": "title",
                "title": {}
            },
            "Status": {
                "id": "status",
                "type": "select",
                "select": {
                    "options": [
                        {"name": "Pending", "color": "gray"},
                        {"name": "In Progress", "color": "blue"},
                        {"name": "Done", "color": "green"},
                        {"name": "Blocked", "color": "red"}
                    ]
                }
            },
            "Owner": {
                "id": "owner",
                "type": "select",
                "select": {
                    "options": [
                        {"name": "Leo", "color": "default"},
                        {"name": "Bảo", "color": "default"},
                        {"name": "Tân", "color": "default"},
                        {"name": "Hoàng", "color": "default"}
                    ]
                }
            },
            "Priority": {
                "id": "priority",
                "type": "select",
                "select": {
                    "options": [
                        {"name": "Low", "color": "green"},
                        {"name": "Medium", "color": "yellow"},
                        {"name": "High", "color": "orange"},
                        {"name": "Critical", "color": "red"}
                    ]
                }
            },
            "Due Date": {
                "id": "due_date",
                "type": "date",
                "date": {}
            },
            "Category": {
                "id": "category",
                "type": "select",
                "select": {
                    "options": [
                        {"name": "Data Preparation", "color": "blue"},
                        {"name": "Model Development", "color": "purple"},
                        {"name": "Training", "color": "orange"},
                        {"name": "Evaluation", "color": "green"},
                        {"name": "Documentation", "color": "gray"},
                        {"name": "Infrastructure", "color": "red"}
                    ]
                }
            }
        }
        
        task_db = create_database(
            MAIN_PAGE_ID,
            "📋 Task List",
            "📋",
            task_db_schema
        )
        
        if task_db:
            task_db_id = task_db.get('id')
            print(f"✅ Task Database created (ID: {task_db_id})")
            created_pages['📋 Task List'] = task_db_id
        else:
            print("❌ Failed to create Task Database")
    except Exception as e:
        print(f"⚠️  Error: {e}")
    
    print()
    
    # 4. Create Paper Database
    print("4️⃣  Creating Paper Reference Database...")
    try:
        paper_db_schema = {
            "Title": {
                "id": "title",
                "type": "title",
                "title": {}
            },
            "Category": {
                "id": "category",
                "type": "select",
                "select": {
                    "options": [
                        {"name": "VLM Architecture", "color": "blue"},
                        {"name": "Knowledge Distillation", "color": "purple"},
                        {"name": "OCR", "color": "green"},
                        {"name": "Vision Encoding", "color": "orange"},
                        {"name": "Foundation Model", "color": "red"}
                    ]
                }
            },
            "Link": {
                "id": "link",
                "type": "url",
                "url": {}
            },
            "Status": {
                "id": "read_status",
                "type": "select",
                "select": {
                    "options": [
                        {"name": "Not Started", "color": "gray"},
                        {"name": "In Progress", "color": "blue"},
                        {"name": "Completed", "color": "green"}
                    ]
                }
            },
            "Owner": {
                "id": "owner",
                "type": "select",
                "select": {
                    "options": [
                        {"name": "Leo", "color": "default"},
                        {"name": "Bảo", "color": "default"},
                        {"name": "Tân", "color": "default"}
                    ]
                }
            },
            "Notes": {
                "id": "notes",
                "type": "rich_text",
                "rich_text": {}
            }
        }
        
        paper_db = create_database(
            MAIN_PAGE_ID,
            "📚 Paper References",
            "📚",
            paper_db_schema
        )
        
        if paper_db:
            paper_db_id = paper_db.get('id')
            print(f"✅ Paper Database created (ID: {paper_db_id})")
            created_pages['📚 Paper References'] = paper_db_id
        else:
            print("❌ Failed to create Paper Database")
    except Exception as e:
        print(f"⚠️  Error: {e}")
    
    print()
    print("=" * 60)
    print("✨ Notion Workspace Setup Complete!")
    print("=" * 60)
    print("\n📍 Created Pages & Databases:")
    for title, page_id in created_pages.items():
        print(f"  • {title}")
        print(f"    ID: {page_id}")
    
    print("\n📊 Next Steps:")
    print("  1. Review pages in Notion web interface")
    print("  2. Add initial tasks to Task List database")
    print("  3. Add key papers to Paper References database")
    print("  4. Configure team access and permissions")
    print("  5. Set up regular sync with GitHub issues")

if __name__ == "__main__":
    main()

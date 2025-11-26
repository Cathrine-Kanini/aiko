#!/usr/bin/env python3
"""
CBC Curriculum Ingestion Script
Loads curriculum content into ChromaDB vector store
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Dict, List
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings
from app.core.logging import logger


# Sample CBC Curriculum Content
CBC_CURRICULUM = {
    "grade_7_math": [
        {
            "topic": "Fractions - Addition and Subtraction",
            "strand": "Numbers",
            "sub_strand": "Fractions",
            "learning_outcomes": [
                "Add and subtract fractions with different denominators",
                "Apply fraction operations to solve real-life problems",
                "Understand the concept of equivalent fractions"
            ],
            "content": """
FRACTIONS: ADDITION AND SUBTRACTION

Introduction:
When adding or subtracting fractions with different denominators, we must first find a common denominator.

Steps for Adding/Subtracting Fractions:
1. Find the Least Common Multiple (LCM) of the denominators
2. Convert each fraction to an equivalent fraction with the common denominator
3. Add or subtract the numerators
4. Keep the denominator the same
5. Simplify the answer if possible

Example 1: Add 1/3 + 1/4
Step 1: LCM of 3 and 4 is 12
Step 2: Convert to equivalent fractions
  1/3 = 4/12 (multiply numerator and denominator by 4)
  1/4 = 3/12 (multiply numerator and denominator by 3)
Step 3: Add numerators: 4 + 3 = 7
Step 4: Result: 7/12

Example 2: Subtract 3/5 - 1/3
Step 1: LCM of 5 and 3 is 15
Step 2: Convert to equivalent fractions
  3/5 = 9/15
  1/3 = 5/15
Step 3: Subtract numerators: 9 - 5 = 4
Step 4: Result: 4/15

Practice Problems:
1. 1/2 + 1/3 = ?
2. 2/3 - 1/4 = ?
3. 3/4 + 2/5 = ?

Real-Life Applications:
- Cooking: Adding 1/2 cup of flour and 1/3 cup of sugar
- Sharing: If you eat 1/4 of a pizza and your friend eats 1/3, how much is eaten?
- Money: Combining fractions of money amounts
            """,
            "key_terms": ["fraction", "numerator", "denominator", "LCM", "equivalent fractions", "common denominator"],
            "key_concepts": [
                "Fractions represent parts of a whole",
                "Different denominators need a common denominator for addition/subtraction",
                "LCM helps find the smallest common denominator"
            ]
        },
        {
            "topic": "Decimals - Converting and Comparing",
            "strand": "Numbers",
            "sub_strand": "Decimals",
            "learning_outcomes": [
                "Convert fractions to decimals and vice versa",
                "Compare and order decimal numbers",
                "Round decimals to given places"
            ],
            "content": """
DECIMALS: CONVERTING AND COMPARING

What are Decimals?
A decimal is another way to write a fraction. The decimal point separates whole numbers from parts of a whole.

Converting Fractions to Decimals:
Method 1: Divide the numerator by the denominator
Example: 3/4 = 3 ÷ 4 = 0.75

Method 2: Convert to denominator of 10, 100, or 1000
Example: 1/2 = 5/10 = 0.5

Common Fraction-Decimal Equivalents to Remember:
1/2 = 0.5
1/4 = 0.25
3/4 = 0.75
1/5 = 0.2
1/10 = 0.1
1/100 = 0.01

Comparing Decimals:
Compare digits from left to right, starting after the decimal point.

Example 1: Which is larger: 0.45 or 0.5?
0.45 = 0.450
0.5 = 0.500
Compare tenths place: 4 < 5
Therefore: 0.45 < 0.5

Place Value in Decimals:
- Tenths: first digit after decimal point (0.1)
- Hundredths: second digit (0.01)
- Thousandths: third digit (0.001)

Real-Life Applications:
- Money: KES 50.75 (50 shillings and 75 cents)
- Measurements: 1.5 meters, 2.3 kilograms
- Prices in shops: Everything uses decimals
- Sports: Race times (10.5 seconds)
            """,
            "key_terms": ["decimal", "decimal point", "place value", "tenths", "hundredths", "thousandths"],
            "key_concepts": [
                "Decimals are another way to represent fractions",
                "Place value is crucial in comparing decimals",
                "Decimals are used extensively in money and measurements"
            ]
        },
        {
            "topic": "Percentages - Understanding and Calculating",
            "strand": "Numbers",
            "sub_strand": "Percentages",
            "learning_outcomes": [
                "Convert between fractions, decimals, and percentages",
                "Calculate percentages of quantities",
                "Solve problems involving percentage increase and decrease"
            ],
            "content": """
PERCENTAGES: UNDERSTANDING AND CALCULATING

What is a Percentage?
Percentage means 'per hundred' or 'out of 100'. The symbol is %.

Converting to Percentages:
From Fraction: Multiply by 100
Example: 3/4 = (3/4) × 100 = 75%

From Decimal: Multiply by 100
Example: 0.65 = 0.65 × 100 = 65%

Converting from Percentages:
To Decimal: Divide by 100
Example: 45% = 45 ÷ 100 = 0.45

To Fraction: Write over 100 and simplify
Example: 25% = 25/100 = 1/4

Finding a Percentage of a Quantity:
Method 1: Convert to decimal, then multiply
Example: Find 25% of 80
25% = 0.25
0.25 × 80 = 20

Method 2: Use fraction
25% = 1/4
1/4 × 80 = 20

Common Percentages to Remember:
50% = 1/2 = 0.5
25% = 1/4 = 0.25
75% = 3/4 = 0.75
10% = 1/10 = 0.1
100% = 1 (the whole)

Percentage Increase/Decrease:
To find percentage change:
Change = (New Value - Original Value) / Original Value × 100%

Real-Life Applications:
- Shop discounts: "20% off original price"
- Exam scores: "Scored 85% in the test"
- Phone battery: "Battery at 45%"
- Population growth: "Population increased by 3%"
- Interest rates: "Bank offers 5% interest"
            """,
            "key_terms": ["percentage", "percent", "discount", "increase", "decrease", "interest"],
            "key_concepts": [
                "Percent means out of 100",
                "Percentages, fractions, and decimals are related",
                "Percentages are used everywhere in daily life"
            ]
        },
        {
            "topic": "Simple Equations - Solving for Unknowns",
            "strand": "Algebra",
            "sub_strand": "Linear Equations",
            "learning_outcomes": [
                "Solve simple linear equations",
                "Use algebra to solve word problems",
                "Understand the concept of a variable"
            ],
            "content": """
SIMPLE EQUATIONS: SOLVING FOR UNKNOWNS

What is an Equation?
An equation is a mathematical statement showing two expressions are equal.
Example: x + 5 = 12

The unknown value (x) is called a variable.

Golden Rule of Equations:
Whatever you do to one side, you MUST do to the other side!

Solving Equations:

Type 1: Addition/Subtraction
Example: x + 7 = 15
Solution: Subtract 7 from both sides
x + 7 - 7 = 15 - 7
x = 8

Check: 8 + 7 = 15 ✓

Type 2: Multiplication/Division
Example: 3x = 21
Solution: Divide both sides by 3
3x ÷ 3 = 21 ÷ 3
x = 7

Check: 3 × 7 = 21 ✓

Type 3: Two-Step Equations
Example: 2x + 5 = 13
Step 1: Subtract 5 from both sides
2x + 5 - 5 = 13 - 5
2x = 8
Step 2: Divide both sides by 2
2x ÷ 2 = 8 ÷ 2
x = 4

Check: 2(4) + 5 = 8 + 5 = 13 ✓

Word Problems:
Example: John has some books. Mary has 5 more books than John. Together they have 15 books. How many books does John have?

Let x = number of John's books
Then Mary has x + 5 books
Equation: x + (x + 5) = 15
Simplify: 2x + 5 = 15
Solve: 2x = 10
Therefore: x = 5

John has 5 books, Mary has 10 books.

Real-Life Applications:
- Finding unknown quantities (ages, amounts, distances)
- Money problems (savings, spending)
- Sharing problems (dividing items equally)
            """,
            "key_terms": ["equation", "variable", "unknown", "solve", "isolate", "balance"],
            "key_concepts": [
                "Equations show equality between expressions",
                "We solve for unknowns by isolating the variable",
                "What we do to one side, we must do to the other"
            ]
        }
    ],
    
    "grade_7_science": [
        {
            "topic": "Classification of Living Things",
            "strand": "Living Things and their Environment",
            "sub_strand": "Classification",
            "learning_outcomes": [
                "Classify living things into their correct groups",
                "Understand the characteristics of different kingdoms",
                "Explain the importance of classification"
            ],
            "content": """
CLASSIFICATION OF LIVING THINGS

Why Classify Living Things?
Scientists classify living things to:
- Organize the huge diversity of life
- Make it easier to study organisms
- Show relationships between different organisms

The Five Kingdoms:

1. ANIMALS (Animalia)
Characteristics:
- Move from place to place
- Feed on other organisms (cannot make own food)
- Most have sense organs
- Made of many cells
Examples: humans, lions, birds, fish, insects

2. PLANTS (Plantae)
Characteristics:
- Make their own food through photosynthesis
- Have roots, stems, and leaves
- Cannot move from place to place (stay in one spot)
- Most are green (contain chlorophyll)
Examples: trees, grass, flowers, vegetables

3. FUNGI
Characteristics:
- Feed on dead or decaying matter
- Do not have chlorophyll (cannot make own food)
- Can be single-celled or multi-celled
Examples: mushrooms, molds, yeast

4. PROTISTS
Characteristics:
- Mostly single-celled organisms
- Live in water
- Some can make food, others cannot
Examples: amoeba, paramecium, algae

5. BACTERIA (Monera)
Characteristics:
- Very tiny, single-celled organisms
- Can only be seen with a microscope
- Found everywhere (soil, water, air, inside organisms)
Examples: E. coli, beneficial bacteria in yogurt

Characteristics Used for Classification:
- How they move
- How they feed (make food or eat others)
- How they reproduce
- Body structure (cells, organs)
- Habitat (where they live)

Classification Levels (from largest to smallest):
Kingdom → Phylum → Class → Order → Family → Genus → Species

Example: Human Classification
Kingdom: Animalia
Phylum: Chordata
Class: Mammalia
Order: Primates
Family: Hominidae
Genus: Homo
Species: Homo sapiens

Real-Life Applications:
- Farming: Knowing which plants grow best together
- Medicine: Understanding which organisms cause diseases
- Conservation: Protecting endangered species
- Food production: Using bacteria for yogurt, yeast for bread
            """,
            "key_terms": ["classification", "kingdom", "species", "characteristics", "organisms", "phylum"],
            "key_concepts": [
                "Living things are organized into five kingdoms",
                "Classification helps us study and understand life",
                "Organisms in the same group share similar characteristics"
            ]
        }
    ]
}

def create_documents_from_curriculum(curriculum_data: Dict) -> List[Document]:
    """Convert curriculum dictionary to LangChain Documents"""
    documents = []
    
    for subject_key, topics in curriculum_data.items():
        # Parse subject and grade
        parts = subject_key.split('_')
        grade = parts[1]
        subject = ' '.join(parts[2:]).lower()
        
        for topic_data in topics:
            # Create comprehensive document
            content = f"""SUBJECT: {subject.title()}
GRADE: {grade}
TOPIC: {topic_data['topic']}
STRAND: {topic_data['strand']}
SUB-STRAND: {topic_data.get('sub_strand', 'General')}

LEARNING OUTCOMES:
{chr(10).join(['- ' + outcome for outcome in topic_data['learning_outcomes']])}

CURRICULUM CONTENT:
{topic_data['content']}

KEY TERMS: {', '.join(topic_data.get('key_terms', []))}

KEY CONCEPTS:
{chr(10).join(['- ' + concept for concept in topic_data.get('key_concepts', [])])}
"""
            
            doc = Document(
                page_content=content,
                metadata={
                    "grade": grade,
                    "subject": subject,
                    "topic": topic_data['topic'],
                    "strand": topic_data['strand'],
                    "sub_strand": topic_data.get('sub_strand', 'General'),
                    "type": "curriculum_content",
                    "language": "en",
                    "num_outcomes": len(topic_data['learning_outcomes'])
                }
            )
            documents.append(doc)
    
    return documents

def ingest_curriculum():
    """Main ingestion function"""
    print("=" * 60)
    print("🚀 CBC CURRICULUM INGESTION")
    print("=" * 60)
    
    # Initialize embeddings
    print("\n📊 Step 1: Loading embedding model...")
    print(f"   Model: {settings.embedding_model}")
    embeddings = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={'device': 'cpu'}
    )
    print("   ✅ Embedding model loaded successfully")
    
    # Create documents
    print("\n📚 Step 2: Creating documents from curriculum...")
    documents = create_documents_from_curriculum(CBC_CURRICULUM)
    print(f"   ✅ Created {len(documents)} documents")
    
    # Display summary
    print("\n📋 Document Summary:")
    for doc in documents:
        print(f"   - {doc.metadata['topic']} (Grade {doc.metadata['grade']}, {doc.metadata['subject'].title()})")
    
    # Split documents
    print("\n✂️  Step 3: Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    split_docs = text_splitter.split_documents(documents)
    print(f"   ✅ Created {len(split_docs)} chunks")
    
    # Create vector store
    print("\n💾 Step 4: Creating vector store...")
    print(f"   Location: {settings.chroma_persist_dir}")
    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=settings.chroma_persist_dir,
        collection_name="cbc_curriculum"
    )
    
    # No need to call persist() in Chroma 0.4+
    print("   ✅ Vector store created and persisted")
    
    # Test the store
    print("\n🔍 Step 5: Testing vector store...")
    test_queries = [
        ("How do I add fractions?", "7", "math"),
        ("What is classification?", "7", "science"),
    ]
    
    for query, grade, subject in test_queries:
        print(f"\n   Test Query: '{query}'")
        print(f"   Grade: {grade}, Subject: {subject}")
        
        # Fixed filter syntax for ChromaDB
        results = vectorstore.similarity_search(
            query,
            k=2,
            filter={
                "$and": [
                    {"grade": {"$eq": grade}},
                    {"subject": {"$eq": subject}}
                ]
            }
        )
        
        print(f"   Found: {len(results)} relevant chunks")
        if results:
            print(f"   Top result preview:")
            print(f"   {results[0].page_content[:150]}...")
    
    # Final summary
    print("\n" + "=" * 60)
    print("✅ INGESTION COMPLETE!")
    print("=" * 60)
    print(f"📊 Statistics:")
    print(f"   - Documents created: {len(documents)}")
    print(f"   - Chunks stored: {len(split_docs)}")
    print(f"   - Database location: {settings.chroma_persist_dir}")
    print(f"   - Collection: cbc_curriculum")
    print("\n🚀 Ready to start API server!")
    print("   Run: uvicorn app.main:app --reload")
    print("=" * 60)

if __name__ == "__main__":
    # Check if database already exists
    if os.path.exists(settings.chroma_persist_dir):
        print(f"\n⚠️  Warning: Database already exists at {settings.chroma_persist_dir}")
        response = input("   Do you want to overwrite it? (yes/no): ")
        
        if response.lower() not in ['yes', 'y']:
            print("❌ Ingestion cancelled")
            sys.exit(0)
        
        # Remove existing database
        import shutil
        shutil.rmtree(settings.chroma_persist_dir)
        print("🗑️  Removed existing database")
    
    try:
        ingest_curriculum()
    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        sys.exit(1)
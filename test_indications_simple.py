"""
Simple test for the new extract_indications_only() method
Tests on keytruda.md without needing API calls first (uses regex fallback)
"""

import os
from mit_style_rlm_pharma import PharmaceuticalRLM


def test_indications_extraction_simple():
    """Test the specialized INDICATIONS extraction method"""
    
    print("\n" + "="*80)
    print("🧪 TESTING: extract_indications_only() Method")
    print("="*80 + "\n")
    
    # Load keytruda.md
    try:
        with open("keytruda.md", "r", encoding="utf-8") as f:
            keytruda_text = f.read()
        print(f"✅ Loaded keytruda.md: {len(keytruda_text)} characters\n")
    except FileNotFoundError:
        print("❌ keytruda.md not found!")
        return
    
    # Get API key
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    
    if not DEEPSEEK_API_KEY:
        print("⚠️  DEEPSEEK_API_KEY not set")
        ask = input("Enter your DeepSeek API key (or press Enter to skip RLM test): ")
        if ask.strip():
            DEEPSEEK_API_KEY = ask
            os.environ["DEEPSEEK_API_KEY"] = ask
        else:
            print("\nℹ️  Skipping RLM test, will only test regex fallback\n")
            # Test just the regex method
            from improved_extraction_methods import extract_indications_and_usage_fda
            
            print("="*80)
            print("📋 REGEX EXTRACTION TEST (No API needed)")
            print("="*80 + "\n")
            
            indications = extract_indications_and_usage_fda(keytruda_text)
            
            if indications:
                print("✅ Successfully extracted INDICATIONS section!")
                print(f"\nLength: {len(indications)} characters")
                print("\nFirst 800 characters:")
                print("-" * 80)
                # Print ASCII-safe version
                safe_text = indications[:800].encode('ascii', 'ignore').decode('ascii')
                print(safe_text)
                print("-" * 80)
                
                # Count indications
                lines = [line.strip() for line in indications.split('\n') if line.strip()]
                indication_lines = [line for line in lines if line.startswith('-')]
                print(f"\n📊 Found approximately {len(indication_lines)} bulleted indications")
            else:
                print("❌ Failed to extract INDICATIONS section")
            
            return
    
    # Initialize processor with API
    print("🚀 Initializing MIT-Style RLM Processor...\n")
    processor = PharmaceuticalRLM(DEEPSEEK_API_KEY)
    
    print("="*80)
    print("🎯 CALLING: processor.extract_indications()")
    print("="*80)
    
    # Use the new specialized method
    indications = processor.extract_indications(keytruda_text)
    
    # Display results
    print("\n" + "="*80)
    print("✅ EXTRACTION COMPLETE")
    print("="*80)
    
    print(f"\nTotal length: {len(indications)} characters")
    print("\nFirst 1000 characters:")
    print("-" * 80)
    # Print ASCII-safe version for Windows console
    safe_text = indications[:1000].encode('ascii', 'ignore').decode('ascii')
    print(safe_text)
    print("-" * 80)
    
    # Save to file
    output_file = "keytruda_indications_extracted.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(indications)
    print(f"\n💾 Full extraction saved to: {output_file}")
    
    # Parse indications
    print("\n" + "="*80)
    print("📊 ANALYZING EXTRACTED INDICATIONS")
    print("="*80 + "\n")
    
    lines = [line.strip() for line in indications.split('\n') if line.strip()]
    
    # Count disease categories (capitalized headers)
    diseases = []
    for line in lines:
        # Check if line looks like a disease header
        if line and not line.startswith('-') and not line.startswith('('):
            # Likely a disease category
            if len(line) < 100 and any(c.isupper() for c in line):
                diseases.append(line)
    
    print(f"Disease categories found: {len(diseases)}")
    for i, disease in enumerate(diseases[:10], 1):  # Show first 10
        print(f"  {i}. {disease}")
    
    # Count bulleted indications
    indication_bullets = [line for line in lines if line.startswith('-')]
    print(f"\nBulleted indication lines: {len(indication_bullets)}")

    #save the file
    with open("keytruda_indications_extracted.txt", "w", encoding="utf-8") as f:
        f.write(indications)
    
    print("\n" + "="*80)
    print("✅ TEST COMPLETE")
    print("="*80)


if __name__ == "__main__":
    test_indications_extraction_simple()

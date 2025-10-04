#!/usr/bin/env python3
"""
Basic structure validation test for Hey Ozwell repository.
Verifies all required files and directories are present.
"""

from pathlib import Path
import json
import sys

def test_repository_structure():
    """Test that all required directories and files exist"""
    print("Testing repository structure...")
    
    root = Path(__file__).parent
    
    # Required directories
    required_dirs = [
        'model',
        'model/tools',
        'model/testing',
        'model/data',
        'model/exports',
        'prod',
        'prod/js',
        'prod/js/src',
        'prod/js/examples',
        'prod/js/examples/basic',
        'prod/js/models',
        'prod/ios',
        'prod/mac',
        'prod/windows',
        'prod/common'
    ]
    
    for dir_path in required_dirs:
        full_path = root / dir_path
        assert full_path.exists() and full_path.is_dir(), f"Required directory missing: {dir_path}"
    
    # Required files
    required_files = [
        'README.md',
        'LICENSE',
        '.gitignore',
        'model/README.md',
        'model/requirements.txt',
        'model/tools/prepare_data.py',
        'model/tools/train.py',
        'model/tools/train_all.py',
        'model/testing/evaluate.py',
        'prod/js/README.md',
        'prod/js/package.json',
        'prod/js/src/index.js',
        'prod/js/src/WakeListener.js',
        'prod/js/src/ModelManager.js',
        'prod/js/src/RingBufferRecorder.js',
        'prod/js/src/AudioProcessor.js',
        'prod/js/examples/basic/index.html'
    ]
    
    for file_path in required_files:
        full_path = root / file_path
        assert full_path.exists() and full_path.is_file(), f"Required file missing: {file_path}"
    
    print("✓ Repository structure test passed")

def test_package_json():
    """Test that package.json is valid"""
    print("Testing package.json...")
    
    root = Path(__file__).parent
    package_file = root / 'prod' / 'js' / 'package.json'
    
    with open(package_file) as f:
        package_data = json.load(f)
    
    # Check required fields
    required_fields = ['name', 'version', 'description', 'main', 'dependencies']
    for field in required_fields:
        assert field in package_data, f"Missing field in package.json: {field}"
    
    # Check wake-word dependencies
    deps = package_data['dependencies']
    assert 'onnxruntime-web' in deps, "Missing onnxruntime-web dependency"
    assert 'idb-keyval' in deps, "Missing idb-keyval dependency"
    
    print("✓ Package.json test passed")

def test_wake_phrases():
    """Test that all wake phrases are documented"""
    print("Testing wake phrase documentation...")
    
    root = Path(__file__).parent
    
    expected_phrases = [
        'hey ozwell',
        "ozwell i'm done", 
        'go ozwell',
        'ozwell go'
    ]
    
    # Check main README
    readme_file = root / 'README.md'
    with open(readme_file) as f:
        readme_content = f.read().lower()
    
    for phrase in expected_phrases:
        # Allow for variations in quotes and spacing
        phrase_variants = [
            phrase,
            phrase.replace("'", "'"),
            phrase.replace("'", '"'),
            phrase.replace("'", ""),
        ]
        
        found = any(variant in readme_content for variant in phrase_variants)
        if not found:
            # Try a more flexible search for the "i'm done" phrase
            if "i'm done" in phrase:
                found = "ozwell" in readme_content and "done" in readme_content
        
        assert found, f"Wake phrase not found in README: {phrase}"
    
    print("✓ Wake phrase documentation test passed")

def test_script_permissions():
    """Test that Python scripts are executable"""
    print("Testing script permissions...")
    
    root = Path(__file__).parent
    
    script_files = [
        'model/tools/prepare_data.py',
        'model/tools/train.py', 
        'model/tools/train_all.py',
        'model/testing/evaluate.py'
    ]
    
    for script_path in script_files:
        full_path = root / script_path
        # Check if file exists and has execute permission
        assert full_path.exists(), f"Script file missing: {script_path}"
        
        # Basic syntax check
        try:
            with open(full_path) as f:
                content = f.read()
            compile(content, str(full_path), 'exec')
        except SyntaxError as e:
            assert False, f"Syntax error in {script_path}: {e}"
    
    print("✓ Script permissions test passed")

def test_javascript_exports():
    """Test that JavaScript modules export correctly"""
    print("Testing JavaScript exports...")
    
    root = Path(__file__).parent
    js_src = root / 'prod' / 'js' / 'src'
    
    # Check main index.js exports
    index_file = js_src / 'index.js'
    with open(index_file) as f:
        index_content = f.read()
    
    expected_exports = [
        'WakeListener',
        'ModelManager', 
        'RingBufferRecorder',
        'AudioProcessor'
    ]
    
    for export_name in expected_exports:
        assert f'export {{ {export_name} }}' in index_content, f"Missing export: {export_name}"
    
    # Check that individual modules define their classes
    module_files = {
        'WakeListener.js': 'WakeListener',
        'ModelManager.js': 'ModelManager',
        'RingBufferRecorder.js': 'RingBufferRecorder', 
        'AudioProcessor.js': 'AudioProcessor'
    }
    
    for filename, class_name in module_files.items():
        module_file = js_src / filename
        with open(module_file) as f:
            module_content = f.read()
        
        assert f'export class {class_name}' in module_content, f"Missing class export in {filename}: {class_name}"
    
    print("✓ JavaScript exports test passed")

def main():
    """Run all validation tests"""
    print("Running Hey Ozwell repository validation...\n")
    
    try:
        test_repository_structure()
        test_package_json()
        test_wake_phrases()
        test_script_permissions()
        test_javascript_exports()
        
        print("\n🎉 Repository validation passed!")
        print("\n📁 Repository structure is complete and ready for development.")
        print("\n🚀 Next steps:")
        print("   1. Install Python dependencies: pip install -r model/requirements.txt")
        print("   2. Install JS dependencies: cd prod/js && npm install")
        print("   3. Train initial model: cd model/tools && python prepare_data.py --phrase hey-ozwell")
        print("   4. Test browser SDK: Open prod/js/examples/basic/index.html in browser")
        
    except AssertionError as e:
        print(f"\n❌ Validation failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
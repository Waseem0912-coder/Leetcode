# @/extractor.py

import os
import re
import argparse
from pathlib import Path
from typing import List, Dict, Set, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class TestCaseExtractor:
    """Extract test cases from Android CTS source code."""
    
    # Test-related annotations
    TEST_ANNOTATIONS = [
        r'@Test\b',
        r'@SmallTest\b',
        r'@MediumTest\b',
        r'@LargeTest\b',
        r'@FlakyTest\b',
        r'@Presubmit\b',
        r'@AppModeFull\b',
        r'@AppModeInstant\b',
    ]
    
    # Patterns for test methods
    JAVA_TEST_METHOD = re.compile(
        r'@\w*[Tt]est\w*.*?\n\s*public\s+void\s+(\w+)\s*\(',
        re.DOTALL
    )
    
    KOTLIN_TEST_METHOD = re.compile(
        r'@\w*[Tt]est\w*.*?\n\s*fun\s+(\w+)\s*\(',
        re.DOTALL
    )
    
    PYTHON_TEST_METHOD = re.compile(
        r'def\s+(test_\w+)\s*\(',
        re.MULTILINE
    )
    
    # Package declaration patterns
    PACKAGE_PATTERN = re.compile(r'package\s+([\w.]+)\s*;')
    
    # Folders to search
    TARGET_FOLDERS = ['tests', 'hostsidetests', 'common']
    
    def __init__(self, root_path: str):
        self.root_path = Path(root_path)
        self.test_cases = []
        
    def extract_all(self) -> List[str]:
        """Extract all test cases from the CTS source."""
        logger.info(f"Scanning CTS directory: {self.root_path}")
        
        for target_folder in self.TARGET_FOLDERS:
            folder_path = self.root_path / target_folder
            if folder_path.exists():
                logger.info(f"Processing folder: {target_folder}")
                self._scan_directory(folder_path)
            else:
                logger.warning(f"Folder not found: {folder_path}")
        
        logger.info(f"Total test cases found: {len(self.test_cases)}")
        return sorted(set(self.test_cases))
    
    def _scan_directory(self, directory: Path):
        """Recursively scan directory for test files."""
        # Find all test files
        test_files = []
        
        # Strategy 1: Look for src/android/*/cts pattern
        test_files.extend(self._find_files_pattern1(directory))
        
        # Strategy 2: Look for any *Test.java, *Test.kt files
        test_files.extend(self._find_files_pattern2(directory))
        
        # Strategy 3: Look in specific test source directories
        test_files.extend(self._find_files_pattern3(directory))
        
        # Remove duplicates
        test_files = list(set(test_files))
        
        logger.info(f"Found {len(test_files)} test files in {directory.name}")
        
        # Process each test file
        for test_file in test_files:
            self._process_test_file(test_file)
    
    def _find_files_pattern1(self, directory: Path) -> List[Path]:
        """Find files matching src/android/*/cts pattern."""
        files = []
        src_dirs = directory.rglob('src/android')
        
        for src_dir in src_dirs:
            # Look for cts folder
            for cts_dir in src_dir.rglob('cts'):
                if cts_dir.is_dir():
                    files.extend(cts_dir.rglob('*Test.java'))
                    files.extend(cts_dir.rglob('*Test.kt'))
                    files.extend(cts_dir.rglob('*test*.py'))
        
        return files
    
    def _find_files_pattern2(self, directory: Path) -> List[Path]:
        """Find all test files recursively."""
        files = []
        
        # Java test files
        files.extend(directory.rglob('*Test.java'))
        files.extend(directory.rglob('*Tests.java'))
        
        # Kotlin test files
        files.extend(directory.rglob('*Test.kt'))
        files.extend(directory.rglob('*Tests.kt'))
        
        # Python test files
        files.extend(directory.rglob('test_*.py'))
        files.extend(directory.rglob('*_test.py'))
        
        return files
    
    def _find_files_pattern3(self, directory: Path) -> List[Path]:
        """Find files in common test source directories."""
        files = []
        test_source_patterns = [
            'src/test/java',
            'src/androidTest/java',
            'src/main/java',
            'src/test/kotlin',
            'src/androidTest/kotlin',
        ]
        
        for pattern in test_source_patterns:
            for test_dir in directory.rglob(pattern):
                if test_dir.is_dir():
                    files.extend(test_dir.rglob('*Test.java'))
                    files.extend(test_dir.rglob('*Test.kt'))
                    files.extend(test_dir.rglob('*Tests.java'))
                    files.extend(test_dir.rglob('*Tests.kt'))
        
        return files
    
    def _process_test_file(self, file_path: Path):
        """Process a single test file and extract test cases."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Determine file type and extract accordingly
            if file_path.suffix == '.java':
                self._extract_java_tests(file_path, content)
            elif file_path.suffix == '.kt':
                self._extract_kotlin_tests(file_path, content)
            elif file_path.suffix == '.py':
                self._extract_python_tests(file_path, content)
                
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
    
    def _extract_java_tests(self, file_path: Path, content: str):
        """Extract test cases from Java files."""
        # Get package name
        package = self._extract_package(content, file_path)
        
        # Get class name (filename without .java)
        class_name = file_path.stem
        
        # Find all test methods
        test_methods = self._find_java_test_methods(content)
        
        if test_methods:
            logger.debug(f"Found {len(test_methods)} tests in {file_path.name}")
            
            for method in test_methods:
                test_case = f"{package}.{class_name}#{method}"
                self.test_cases.append(test_case)
    
    def _extract_kotlin_tests(self, file_path: Path, content: str):
        """Extract test cases from Kotlin files."""
        # Get package name
        package = self._extract_package(content, file_path)
        
        # Get class name
        class_name = file_path.stem
        
        # Find all test methods
        test_methods = self._find_kotlin_test_methods(content)
        
        if test_methods:
            logger.debug(f"Found {len(test_methods)} tests in {file_path.name}")
            
            for method in test_methods:
                test_case = f"{package}.{class_name}#{method}"
                self.test_cases.append(test_case)
    
    def _extract_python_tests(self, file_path: Path, content: str):
        """Extract test cases from Python files."""
        # For Python, derive package from path
        package = self._derive_package_from_path(file_path)
        
        # Get module name
        module_name = file_path.stem
        
        # Find all test methods
        test_methods = self._find_python_test_methods(content)
        
        if test_methods:
            logger.debug(f"Found {len(test_methods)} tests in {file_path.name}")
            
            for method in test_methods:
                test_case = f"{package}.{module_name}#{method}"
                self.test_cases.append(test_case)
    
    def _find_java_test_methods(self, content: str) -> List[str]:
        """Find Java test method names."""
        methods = []
        
        # Pattern 1: @Test annotation
        matches = self.JAVA_TEST_METHOD.findall(content)
        methods.extend(matches)
        
        # Pattern 2: Methods starting with 'test'
        test_prefix_pattern = re.compile(
            r'public\s+void\s+(test\w+)\s*\(',
            re.MULTILINE
        )
        matches = test_prefix_pattern.findall(content)
        methods.extend(matches)
        
        return list(set(methods))  # Remove duplicates
    
    def _find_kotlin_test_methods(self, content: str) -> List[str]:
        """Find Kotlin test method names."""
        methods = []
        
        # Pattern 1: @Test annotation
        matches = self.KOTLIN_TEST_METHOD.findall(content)
        methods.extend(matches)
        
        # Pattern 2: Functions starting with 'test'
        test_prefix_pattern = re.compile(
            r'fun\s+(test\w+)\s*\(',
            re.MULTILINE
        )
        matches = test_prefix_pattern.findall(content)
        methods.extend(matches)
        
        return list(set(methods))
    
    def _find_python_test_methods(self, content: str) -> List[str]:
        """Find Python test method names."""
        matches = self.PYTHON_TEST_METHOD.findall(content)
        return list(set(matches))
    
    def _extract_package(self, content: str, file_path: Path) -> str:
        """Extract package name from file content or path."""
        # Try to find package declaration in file
        match = self.PACKAGE_PATTERN.search(content)
        if match:
            return match.group(1)
        
        # Fall back to deriving from path
        return self._derive_package_from_path(file_path)
    
    def _derive_package_from_path(self, file_path: Path) -> str:
        """Derive package name from file path."""
        parts = file_path.parts
        
        # Look for 'android' in path and build package from there
        try:
            # Find 'src' directory
            if 'src' in parts:
                src_idx = parts.index('src')
                # Get everything after src until the file
                package_parts = []
                
                for i in range(src_idx + 1, len(parts) - 1):
                    part = parts[i]
                    # Skip common non-package directories
                    if part not in ['java', 'kotlin', 'main', 'test', 'androidTest']:
                        package_parts.append(part)
                
                if package_parts:
                    return '.'.join(package_parts)
            
            # Alternative: look for android in path
            if 'android' in parts:
                android_idx = parts.index('android')
                package_parts = parts[android_idx:-1]
                return '.'.join(package_parts)
                
        except (ValueError, IndexError):
            pass
        
        # Last resort: use file's parent directory
        return file_path.parent.name
    
    def save_results(self, output_file: str):
        """Save extracted test cases to a file."""
        test_cases = self.extract_all()
        
        with open(output_file, 'w') as f:
            for test_case in test_cases:
                f.write(f"{test_case}\n")
        
        logger.info(f"Results saved to {output_file}")
        return len(test_cases)


class AdvancedTestExtractor(TestCaseExtractor):
    """Advanced extractor with more flexible patterns."""
    
    def _find_java_test_methods(self, content: str) -> List[str]:
        """Enhanced Java test method detection."""
        methods = set()
        
        # Remove comments to avoid false positives
        content = self._remove_comments(content)
        
        # Pattern 1: Any @*Test* annotation followed by method
        pattern1 = re.compile(
            r'@\w*[Tt]est\w*[^{]*?\n\s*(?:public\s+)?(?:protected\s+)?void\s+(\w+)\s*\(',
            re.DOTALL
        )
        methods.update(pattern1.findall(content))
        
        # Pattern 2: JUnit 3 style - methods starting with 'test'
        pattern2 = re.compile(
            r'(?:public\s+)?void\s+(test\w+)\s*\(',
            re.MULTILINE
        )
        methods.update(pattern2.findall(content))
        
        # Pattern 3: Methods with @Test annotation (multiline)
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if '@Test' in line or '@test' in line:
                # Look at next few lines for method declaration
                for j in range(i + 1, min(i + 5, len(lines))):
                    method_match = re.search(r'(?:public\s+)?(?:void|fun)\s+(\w+)\s*\(', lines[j])
                    if method_match:
                        methods.add(method_match.group(1))
                        break
        
        return list(methods)
    
    def _remove_comments(self, content: str) -> str:
        """Remove Java/Kotlin comments from content."""
        # Remove single-line comments
        content = re.sub(r'//.*?$', '', content, flags=re.MULTILINE)
        # Remove multi-line comments
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        return content


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Extract test case names from Android CTS source code'
    )
    parser.add_argument(
        'cts_path',
        help='Path to the CTS source code directory'
    )
    parser.add_argument(
        '-o', '--output',
        default='testcases.txt',
        help='Output file for test case names (default: testcases.txt)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--advanced',
        action='store_true',
        help='Use advanced extraction patterns'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Choose extractor
    if args.advanced:
        extractor = AdvancedTestExtractor(args.cts_path)
        logger.info("Using advanced extraction mode")
    else:
        extractor = TestCaseExtractor(args.cts_path)
    
    # Extract and save
    count = extractor.save_results(args.output)
    
    print(f"\n{'='*60}")
    print(f"Extraction Complete!")
    print(f"Total test cases found: {count}")
    print(f"Results saved to: {args.output}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""List all categories used in index.qmd files.

This script scans all index.qmd files in the books repository and extracts
the categories field. It displays each unique category with its usage count,
helping to prevent inconsistent category naming.

Usage:
    python list_categories.py [directory]

    If no directory is specified, uses the current working directory.
    The script will search for all index.qmd files recursively.
"""

import re
import sys
from pathlib import Path
from collections import Counter


def extract_categories_from_file(file_path):
    """Extract categories from a single index.qmd file.

    Args:
        file_path: Path to the index.qmd file

    Returns:
        list: List of category strings found in the file
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Find categories line in YAML frontmatter
        # Pattern: categories: [Category1, Category2, ...]
        match = re.search(r'^categories:\s*\[(.*?)\]', content, re.MULTILINE)

        if match:
            categories_str = match.group(1)
            # Split by comma and strip whitespace
            categories = [cat.strip() for cat in categories_str.split(',')]
            return categories

        return []

    except Exception as e:
        print(f"Warning: Could not read {file_path}: {e}", file=sys.stderr)
        return []


def main():
    """Main function to list all categories."""
    # Get directory from command line argument or use current directory
    if len(sys.argv) > 1:
        base_dir = Path(sys.argv[1])
    else:
        base_dir = Path.cwd()

    if not base_dir.exists():
        print(f"Error: Directory {base_dir} not found!")
        sys.exit(1)

    # Find all index.qmd files
    index_files = list(base_dir.rglob('*/index.qmd'))

    if not index_files:
        print(f"No index.qmd files found in {base_dir}")
        sys.exit(0)

    print(f"Found {len(index_files)} index.qmd file(s) in {base_dir}\n")

    # Collect all categories
    all_categories = []
    file_categories = {}

    for index_file in sorted(index_files):
        categories = extract_categories_from_file(index_file)
        if categories:
            # Store relative path for display
            rel_path = index_file.relative_to(base_dir)
            file_categories[rel_path] = categories
            all_categories.extend(categories)

    if not all_categories:
        print("No categories found in any index.qmd files")
        sys.exit(0)

    # Count category usage
    category_counts = Counter(all_categories)

    # Display results
    print("=" * 60)
    print("CATEGORY USAGE")
    print("=" * 60)

    for category, count in category_counts.most_common():
        print(f"{category:<40} ({count})")

    print("\n" + "=" * 60)
    print("CATEGORIES BY FILE")
    print("=" * 60)

    for file_path, categories in sorted(file_categories.items()):
        print(f"\n{file_path}:")
        for cat in categories:
            print(f"  - {cat}")

    print("\n" + "=" * 60)
    print(f"Total unique categories: {len(category_counts)}")
    print(f"Total category instances: {len(all_categories)}")
    print("=" * 60)


if __name__ == '__main__':
    main()

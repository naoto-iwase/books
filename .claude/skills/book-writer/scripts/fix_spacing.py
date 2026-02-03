#!/usr/bin/env python3
"""Fix list and blockquote spacing in qmd files.

This script ensures that all lists and blockquotes are preceded by a blank line,
as required by the book-writer skill formatting rules.

It uses the following regex patterns:

1. List spacing:
    ^(?!\s*(?:[-*+]|\d+\.) )(?=.*\S)(.+)\n((?:[-*+]|\d+\.) )
    Replacement: $1\n\n$2

2. Blockquote spacing (before blockquote blocks):
    ^(?!\s*>)(.*\S)\n(\s*>.*)
    Replacement: $1\n\n$2

This finds lines that are directly followed by a list item or blockquote block
(without a blank line) and inserts a blank line between them.

Usage:
    python fix_spacing.py <directory>

    Processes all .qmd files in the specified directory.
"""

import re
import sys
from pathlib import Path


def fix_list_spacing(content):
    """Fix spacing before list items and blockquotes in content.

    Args:
        content: String containing markdown content

    Returns:
        str: Content with fixed list and blockquote spacing
    """
    # Pattern 1: Match lines followed directly by list items (without blank line)
    # ^(?!\s*(?:[-*+]|\d+\.) )  - Line doesn't start with list marker (with optional whitespace)
    #                             Supports: -, *, +, or numbered lists (1., 2., etc.)
    # (?=.*\S)                   - Line contains non-whitespace characters
    # (.+)                       - Capture the line content
    # \n                         - Followed by newline
    # ((?:[-*+]|\d+\.) )         - Followed by list marker with space
    list_pattern = r'^(?!\s*(?:[-*+]|\d+\.) )(?=.*\S)(.+)\n((?:[-*+]|\d+\.) )'

    # Pattern 2: Match lines followed directly by blockquote blocks (without blank line)
    # ^(?!\s*>)                  - Line doesn't start with > (blockquote marker)
    # (.*\S)                     - Capture line with non-whitespace content
    # \n                         - Followed by newline
    # (\s*>.*)                   - Followed by line starting with > (possibly with indentation)
    blockquote_pattern = r'^(?!\s*>)(.*\S)\n(\s*>.*)'

    # Replace with the line, two newlines (creating blank line), then the marker
    replacement = r'\1\n\n\2'

    # Use MULTILINE flag so ^ matches start of each line
    fixed_content = re.sub(list_pattern, replacement, content, flags=re.MULTILINE)
    fixed_content = re.sub(blockquote_pattern, replacement, fixed_content, flags=re.MULTILINE)

    return fixed_content


def process_file(file_path):
    """Process a single qmd file to fix list spacing.

    Args:
        file_path: Path to the qmd file

    Returns:
        bool: True if file was modified, False otherwise
    """
    print(f"Processing {file_path}...")

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    fixed_content = fix_list_spacing(content)

    if content != fixed_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        print(f"  ✓ Fixed {file_path}")
        return True
    else:
        print(f"  - No changes needed for {file_path}")
        return False


def main():
    """Main function to process qmd files."""
    # Get directory from command line argument
    if len(sys.argv) > 1:
        target_dir = Path(sys.argv[1])
    else:
        print("Usage: python fix_spacing.py <directory>")
        print("Error: Please specify a directory to process")
        sys.exit(1)

    if not target_dir.exists():
        print(f"Directory {target_dir} not found!")
        sys.exit(1)

    qmd_files = list(target_dir.glob('*.qmd'))
    print(f"Found {len(qmd_files)} qmd files in {target_dir}\n")

    fixed_count = 0
    for qmd_file in sorted(qmd_files):
        if process_file(qmd_file):
            fixed_count += 1

    print(f"\n{'='*60}")
    print(f"Fixed {fixed_count} file(s)")


if __name__ == '__main__':
    main()

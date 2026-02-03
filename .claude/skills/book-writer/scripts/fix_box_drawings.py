#!/usr/bin/env python3
"""Fix box drawings alignment in qmd files.

This script fixes the alignment of ASCII box drawings in Quarto markdown files.
It ensures that the right edge (│) of all box drawings is properly aligned by
calculating the display width of each line (considering full-width characters).

Usage:
    python fix_box_drawings.py <directory>

    Processes all .qmd files in the specified directory.
"""

import re
import sys
import unicodedata
from pathlib import Path


def count_display_width(text):
    """Count display width of text.

    Box drawings should only contain half-width characters (ASCII, box drawing
    characters, arrows, etc.). All characters are treated as width 1.

    If full-width characters (e.g., Japanese) are detected, a warning is printed
    as they violate the formatting rules.

    Args:
        text: String to measure

    Returns:
        int: Display width (all characters = 1)
    """
    width = 0
    has_fullwidth = False
    fullwidth_chars = []

    for char in text:
        width += 1
        # Check East Asian Width property
        # F (Fullwidth) or W (Wide) are full-width characters
        ea_width = unicodedata.east_asian_width(char)
        if ea_width in ('F', 'W'):
            has_fullwidth = True
            fullwidth_chars.append(char)

    if has_fullwidth:
        chars_str = ''.join(fullwidth_chars[:10])  # Show first 10 chars
        print(f"    WARNING: Full-width characters detected: {chars_str}")

    return width


def fix_box_drawing(box_text):
    """Fix alignment of a box drawing block.

    Args:
        box_text: String containing box drawing

    Returns:
        str: Fixed box drawing with aligned right edge
    """
    lines = box_text.split('\n')

    # Find the maximum content width (excluding box characters)
    max_content_width = 0
    for line in lines:
        if line.startswith('│') and line.endswith('│'):
            # Extract content between │ characters
            content = line[1:-1]
            width = count_display_width(content)
            max_content_width = max(max_content_width, width)
        elif line.startswith('├') and line.endswith('┤'):
            # Extract content between ├ and ┤
            content = line[1:-1]
            width = count_display_width(content)
            max_content_width = max(max_content_width, width)

    # Rebuild lines with proper alignment
    fixed_lines = []
    for line in lines:
        if line.startswith('┌') and line.endswith('┐'):
            # Top border
            fixed_lines.append('┌' + '─' * max_content_width + '┐')
        elif line.startswith('└') and line.endswith('┘'):
            # Bottom border
            fixed_lines.append('└' + '─' * max_content_width + '┘')
        elif line.startswith('├') and line.endswith('┤'):
            # Middle separator
            content = line[1:-1].replace('─', '')  # Remove existing dashes
            content = content.strip()
            if content:
                # Has text content
                padding_needed = max_content_width - count_display_width(content)
                fixed_lines.append('├' + content + ' ' * padding_needed + '┤')
            else:
                # Just separator line
                fixed_lines.append('├' + '─' * max_content_width + '┤')
        elif line.startswith('│') and line.endswith('│'):
            # Content line
            content = line[1:-1]
            current_width = count_display_width(content)
            padding_needed = max_content_width - current_width
            fixed_lines.append('│' + content + ' ' * padding_needed + '│')
        else:
            # Keep as is (shouldn't happen in a proper box)
            fixed_lines.append(line)

    return '\n'.join(fixed_lines)


def process_file(file_path):
    """Process a single qmd file to fix box drawings.

    Args:
        file_path: Path to the qmd file

    Returns:
        bool: True if file was modified, False otherwise
    """
    print(f"Processing {file_path}...")

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find all box drawing blocks (between ``` markers)
    pattern = r'```\n(┌[─┐]+.*?└[─┘]+)\n```'

    def replace_box(match):
        box_text = match.group(1)
        # Only process if it looks like a box drawing
        if '│' in box_text:
            fixed = fix_box_drawing(box_text)
            return f'```\n{fixed}\n```'
        return match.group(0)

    fixed_content = re.sub(pattern, replace_box, content, flags=re.DOTALL)

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
        print("Usage: python fix_box_drawings.py <directory>")
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

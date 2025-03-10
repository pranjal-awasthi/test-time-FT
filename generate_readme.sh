#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}     README Generator Script            ${NC}"
echo -e "${BLUE}=========================================${NC}"

# Get project name (default to directory name)
PROJECT_NAME=$(basename "$(pwd)")
DESCRIPTION=""
YEAR=$(date +"%Y")
AUTHOR=""
GITHUB_USERNAME=""

# Ask for project information
read -p "Project name [$PROJECT_NAME]: " input
PROJECT_NAME=${input:-$PROJECT_NAME}

read -p "Short description: " DESCRIPTION

read -p "Author name: " AUTHOR

read -p "GitHub username: " GITHUB_USERNAME

# Generate README file
README_FILE="README.md"

echo -e "${GREEN}Generating $README_FILE...${NC}"

# Create README content
cat > "$README_FILE" << EOF
# $PROJECT_NAME

$DESCRIPTION

## Table of Contents
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Contributing](#contributing)
- [License](#license)

## Project Structure
\`\`\`
$(find . -type f -not -path "*/\.*" -not -path "*/venv/*" -not -path "*/__pycache__/*" -not -path "*/\.*/*" | sort | sed 's/\.\///')
\`\`\`

## Installation

This project requires Python and the dependencies listed in \`requirements.txt\`.

### Automatic Setup

Run the setup script to create a virtual environment and install dependencies:

\`\`\`bash
./setup_env.sh
\`\`\`

### Manual Setup

1. Create a virtual environment:
   \`\`\`bash
   python -m venv .venv
   \`\`\`

2. Activate the virtual environment:
   - On Linux/Mac:
     \`\`\`bash
     source .venv/bin/activate
     \`\`\`
   - On Windows:
     \`\`\`bash
     .venv\\Scripts\\activate
     \`\`\`

3. Install dependencies:
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

## Usage

$(if [ -f "main.py" ]; then echo "Run the project with:

\`\`\`bash
python main.py
\`\`\`"; else echo "<!-- Add usage instructions here -->"; fi)

## Features

<!-- List key features of your project -->
- Feature 1
- Feature 2
- Feature 3

## Contributing

1. Fork the repository
2. Create a new branch (\`git checkout -b feature/amazing-feature\`)
3. Make your changes
4. Commit your changes (\`git commit -m 'Add some amazing feature'\`)
5. Push to the branch (\`git push origin feature/amazing-feature\`)
6. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

$AUTHOR - [GitHub](https://github.com/$GITHUB_USERNAME)

EOF

# Detect main Python file if not main.py
if [ ! -f "main.py" ]; then
    POTENTIAL_MAIN_FILES=$(find . -maxdepth 1 -name "*.py" | sort)
    if [ -n "$POTENTIAL_MAIN_FILES" ]; then
        echo -e "${YELLOW}No main.py found. You may want to update the Usage section with one of these Python files:${NC}"
        echo "$POTENTIAL_MAIN_FILES"
    fi
fi

# Look for requirements.txt and setup_env.sh
if [ ! -f "requirements.txt" ]; then
    echo -e "${YELLOW}Warning: requirements.txt not found. Installation instructions may need updating.${NC}"
fi

if [ ! -f "setup_env.sh" ]; then
    echo -e "${YELLOW}Warning: setup_env.sh not found. You may want to create this file or update installation instructions.${NC}"
fi

echo -e "${GREEN}$README_FILE generated successfully!${NC}"
echo -e "${YELLOW}Please review and customize the generated README to better fit your project.${NC}"
echo -e "${BLUE}=========================================${NC}"
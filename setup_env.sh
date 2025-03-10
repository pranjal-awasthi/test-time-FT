#!/bin/bash

# This script is written by Claude 3.7 Sonnet

# Colors for terminal output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Default values
VENV_NAME=".venv"
REQUIREMENTS_FILE="requirements.txt"

# Print header
echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}  Python Virtual Environment Setup Script ${NC}"
echo -e "${BLUE}=========================================${NC}"

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed.${NC}"
    echo "Please install Python 3 and try again."
    exit 1
fi

# Display Python version
PYTHON_VERSION=$(python3 --version)
echo -e "${GREEN}Using ${PYTHON_VERSION}${NC}"

# Check if requirements.txt exists
if [ ! -f "$REQUIREMENTS_FILE" ]; then
    echo -e "${RED}Error: $REQUIREMENTS_FILE not found in the current directory.${NC}"
    exit 1
fi

# Check if virtualenv is installed
if ! command -v python3 -m venv &> /dev/null; then
    echo -e "${YELLOW}Python venv module not found. Installing...${NC}"
    python3 -m pip install --user virtualenv
fi

# Remove existing virtual environment if it exists
if [ -d "$VENV_NAME" ]; then
    echo -e "${YELLOW}Removing existing virtual environment...${NC}"
    rm -rf "$VENV_NAME"
fi

# Create virtual environment
echo -e "${GREEN}Creating virtual environment in ./$VENV_NAME...${NC}"
python3 -m venv "$VENV_NAME"

# Activate virtual environment
echo -e "${GREEN}Activating virtual environment...${NC}"
source "$VENV_NAME/bin/activate"

# Upgrade pip
echo -e "${GREEN}Upgrading pip...${NC}"
pip install --upgrade pip

# Install requirements
echo -e "${GREEN}Installing requirements from $REQUIREMENTS_FILE...${NC}"
pip install -r "$REQUIREMENTS_FILE"

# Check if installation was successful
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Setup complete!${NC}"
    echo -e "${YELLOW}To activate this environment in the future, run:${NC}"
    echo -e "    source $VENV_NAME/bin/activate"
    echo -e "${YELLOW}To deactivate the environment, run:${NC}"
    echo -e "    deactivate"
else
    echo -e "${RED}Error: Failed to install requirements.${NC}"
    exit 1
fi

# List installed packages
echo -e "${GREEN}Installed packages:${NC}"
pip list

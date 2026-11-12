
#!/bin/bash

set -e

packages=(
    "langchain-community"
    "langchain-core"
    "langchain-huggingface"
    "langchain-text-splitters"
)

for package in "${packages[@]}"; do
    echo "Installing $package..."

    if python -m pip install "$package"; then
        echo "Successfully installed $package"
    else
        echo "Failed to install $package. Stopping installation."
        exit 1
    fi

    echo "-----------------------------"
done

echo "All packages installed successfully!"


#!/bin/bash
# setup_ssh_alias.sh
# Setup SSH alias for Humigence client access

echo "🔧 Setting up Humigence SSH Client Access..."

# Get the current directory (where humigencev2 is located)
HUMIGENCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLIENT_SCRIPT="$HUMIGENCE_DIR/client_login.py"

# Make sure the client script is executable
chmod +x "$CLIENT_SCRIPT"

# Create the alias
ALIAS_NAME="humigence-client"
ALIAS_COMMAND="python3 $CLIENT_SCRIPT"

echo "📝 Adding alias to shell configuration..."

# Detect shell and add alias
if [ -n "$ZSH_VERSION" ]; then
    # Zsh
    SHELL_CONFIG="$HOME/.zshrc"
    echo "Detected Zsh shell"
elif [ -n "$BASH_VERSION" ]; then
    # Bash
    SHELL_CONFIG="$HOME/.bashrc"
    echo "Detected Bash shell"
else
    # Default to bash
    SHELL_CONFIG="$HOME/.bashrc"
    echo "Using default Bash shell"
fi

# Check if alias already exists
if grep -q "alias $ALIAS_NAME=" "$SHELL_CONFIG" 2>/dev/null; then
    echo "⚠️  Alias '$ALIAS_NAME' already exists in $SHELL_CONFIG"
    echo "🔄 Updating existing alias..."
    # Remove existing alias
    sed -i "/alias $ALIAS_NAME=/d" "$SHELL_CONFIG"
fi

# Add the alias
echo "alias $ALIAS_NAME='$ALIAS_COMMAND'" >> "$SHELL_CONFIG"

echo "✅ Alias added to $SHELL_CONFIG"
echo ""
echo "🎉 Setup complete!"
echo ""
echo "📋 Usage instructions:"
echo "1. Reload your shell: source $SHELL_CONFIG"
echo "2. SSH into the server: ssh user@server"
echo "3. Run: humigence-client"
echo "4. Enter device name and API key when prompted"
echo ""
echo "🔧 To manage devices, run: python3 $HUMIGENCE_DIR/admin_device_manager.py"
echo ""
echo "💡 The alias will be available after reloading your shell."


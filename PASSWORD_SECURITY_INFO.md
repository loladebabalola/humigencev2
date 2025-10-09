# Password Security Information

## 🔐 API Key Input Behavior

The Humigence client login system is designed with security in mind. Here's how the password input works:

### **In Interactive SSH Sessions (Normal Usage):**
- **API Key characters are HIDDEN** as you type
- **Device Name is visible** (for user convenience)
- Uses `getpass.getpass()` for secure input
- **No characters appear on screen** when typing the API key

### **In Non-Interactive Environments:**
- **API Key characters are VISIBLE** (with warning)
- **Device Name is visible** (as normal)
- Falls back to regular `input()` with warning message
- **Shows warning**: "Using fallback password input (characters will be visible)"

## 🛡️ Security Features

### **Primary Security (Interactive Terminals):**
```
Device Name: iPhone-13
API Key (Password): [characters hidden as you type]
```

### **Fallback Security (Non-Interactive):**
```
Device Name: iPhone-13
⚠️ Using fallback password input (characters will be visible)
API Key (Password) (visible): abc123def456...
```

## 🎯 When You'll See Each Behavior

### **Hidden Characters (Secure):**
- ✅ SSH sessions from terminal
- ✅ Interactive command line usage
- ✅ Most normal usage scenarios
- ✅ When `getpass` works properly

### **Visible Characters (Fallback):**
- ⚠️ Non-interactive environments
- ⚠️ Some automated scripts
- ⚠️ When terminal doesn't support `getpass`
- ⚠️ Remote execution scenarios

## 🔧 How It Works

The system tries multiple methods in order:

1. **`getpass.getpass()`** - Hides characters (preferred)
2. **Fallback with warning** - Shows characters with warning
3. **Ultimate fallback** - Shows characters (last resort)

## 📱 User Experience

### **For Normal Users (SSH + Terminal):**
- Type device name normally
- Type API key with hidden characters
- Secure and user-friendly

### **For Administrators (Scripts/Automation):**
- May see warning about visible characters
- Still functional, just less secure
- Appropriate for automated scenarios

## 🛠️ Technical Implementation

```python
def get_password_input(prompt: str) -> str:
    try:
        # Try getpass first (hides characters)
        return getpass.getpass(prompt)
    except (EOFError, KeyboardInterrupt):
        # Fallback (shows characters with warning)
        console.print("⚠️ Using fallback password input (characters will be visible)")
        return input(f"{prompt} (visible): ")
```

## 🎉 Summary

**In normal SSH usage, your API key characters will be hidden as you type them!** 

The system automatically detects the environment and uses the most secure method available. For regular users connecting via SSH, the experience is secure and user-friendly.

---

**Your API keys are safe! 🔐✨**


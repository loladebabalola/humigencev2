# Paste Functionality Guide

## 📋 Can You Paste API Keys?

**Yes, you can paste API keys!** The system provides multiple ways to input your API key, including paste-friendly options.

## 🔧 How Paste Works

### **Method 1: Hidden Input (Default)**
- **Paste Support**: Depends on terminal
- **Security**: High (characters hidden)
- **Compatibility**: Works in most SSH terminals

### **Method 2: Visible Input (Paste-Friendly)**
- **Paste Support**: ✅ Always works
- **Security**: Lower (characters visible)
- **Compatibility**: Works everywhere

## 🎯 User Experience

When you run `humigence-client`, you'll see:

```
Device Name: iPhone-13

API Key Input:
1. Type manually (hidden characters)
2. Paste from clipboard (visible characters)

Choose input method (1 or 2, default=1): 2
⚠️ Using visible input for pasting (characters will be visible)
API Key (Password) - paste here: [paste your API key here]
```

## 📱 Paste Methods That Work

### **In SSH Terminals:**
- **Ctrl+Shift+V** (Linux)
- **Cmd+V** (Mac)
- **Right-click → Paste**
- **Middle-click** (if enabled)

### **In Terminal Emulators:**
- **Ctrl+Shift+V** (most terminals)
- **Right-click context menu**
- **Edit menu → Paste**

### **In Remote Desktop:**
- **Ctrl+V** (if clipboard sharing enabled)
- **Right-click → Paste**

## 🛡️ Security Considerations

### **Hidden Input (Method 1):**
- ✅ Characters not visible on screen
- ✅ Not visible in terminal history
- ✅ Most secure option
- ⚠️ May not support paste in some terminals

### **Visible Input (Method 2):**
- ⚠️ Characters visible on screen
- ⚠️ May appear in terminal history
- ✅ Always supports paste
- ✅ Good for long/complex API keys

## 🎯 Recommendations

### **For Short API Keys:**
- Use **Method 1** (hidden input)
- Type manually for security

### **For Long/Complex API Keys:**
- Use **Method 2** (visible input)
- Paste from secure source
- Clear terminal history after use

### **For Shared/Public Terminals:**
- Use **Method 1** (hidden input)
- Type manually
- Avoid pasting sensitive data

## 🔄 How the System Works

The system automatically detects your environment and provides the best options:

1. **Tries hidden input first** (most secure)
2. **Falls back to visible input** (if needed)
3. **Gives you the choice** (for paste functionality)

## 📋 Step-by-Step Usage

### **Option 1: Type Manually (Secure)**
```bash
humigence-client
Device Name: iPhone-13
Choose input method (1 or 2, default=1): 1
API Key (Password): [type manually, characters hidden]
```

### **Option 2: Paste from Clipboard (Convenient)**
```bash
humigence-client
Device Name: iPhone-13
Choose input method (1 or 2, default=1): 2
API Key (Password) - paste here: [paste with Ctrl+Shift+V]
```

## 🛠️ Troubleshooting

### **Paste Not Working in Hidden Mode:**
- Choose **Method 2** (visible input)
- Use **Ctrl+Shift+V** instead of **Ctrl+V**
- Try right-click paste

### **Characters Still Visible in Hidden Mode:**
- This is normal in some terminals
- The system will show a warning
- Use **Method 2** for guaranteed paste support

### **Terminal Doesn't Support Paste:**
- Type manually using **Method 1**
- Copy API key to a text file first
- Use a different terminal emulator

## 🎉 Summary

**Yes, you can paste API keys!** The system gives you the choice:

- **Method 1**: Secure typing (may not support paste)
- **Method 2**: Paste-friendly (characters visible)

Choose the method that works best for your situation and terminal environment.

---

**Paste away! 📋✨**


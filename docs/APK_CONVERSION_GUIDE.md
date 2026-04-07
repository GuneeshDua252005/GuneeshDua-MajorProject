## Free APK / Installable App Guide

This project is a Streamlit web application, so the easiest free mobile-install path is not a fully native APK build. The most practical route is:

1. Run the app locally or deploy it to a reachable URL.
2. Open the URL in Chrome on Android.
3. Use **Add to Home Screen**.
4. If a packaged Android app file is needed, use **PWABuilder** on the deployed URL.

### Why this route is recommended

- no paid wrapper is required
- no separate Android UI needs to be written
- the same code works on laptop and phone
- easier for major-project demo and viva

### Important limitation

This is a PWA-style installable experience, not a full offline native Android app.

### If a true native APK is compulsory

You would need a separate packaging stack such as:

- Kivy + Buildozer
- Flutter webview wrapper
- Android WebView app

That would no longer be a strict single-file Streamlit-only solution.

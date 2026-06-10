# Free APK / installable app conversion guide

This project is a Streamlit web application. Streamlit does not directly produce a native Android APK, but you can still create a free installable experience in two practical ways.

## Option 1 - PWA-style install on Windows and Android

This is the easiest and safest approach.

### Windows 11
1. Run or deploy the Streamlit app.
2. Open it in Chrome or Edge.
3. Click the browser install icon in the address bar.
4. Install it as an app.

### Android
1. Open the deployed Streamlit URL in Chrome.
2. Tap the browser menu.
3. Tap **Add to Home Screen** or **Install App**.
4. The app appears like an installed application icon.

This is the recommended free approach for project demonstration.

## Option 2 - Free WebView APK wrapper

If your HOD or evaluator insists on an APK file, use a simple Android Studio WebView wrapper.

### Steps
1. Deploy your Streamlit app publicly.
   - Streamlit Community Cloud
   - Hugging Face Spaces
   - Render
   - Railway

2. Install Android Studio on Windows.

3. Create an Empty Views Activity project.

4. Replace the main activity code with a WebView version:

```kotlin
package com.example.ceiapp

import android.annotation.SuppressLint
import android.os.Bundle
import android.webkit.WebChromeClient
import android.webkit.WebSettings
import android.webkit.WebView
import android.webkit.WebViewClient
import androidx.appcompat.app.AppCompatActivity

class MainActivity : AppCompatActivity() {
    private lateinit var webView: WebView

    @SuppressLint("SetJavaScriptEnabled")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        webView = WebView(this)
        setContentView(webView)

        webView.webViewClient = WebViewClient()
        webView.webChromeClient = WebChromeClient()
        webView.settings.javaScriptEnabled = true
        webView.settings.domStorageEnabled = true
        webView.settings.cacheMode = WebSettings.LOAD_DEFAULT
        webView.loadUrl("https://your-streamlit-app-url")
    }

    override fun onBackPressed() {
        if (webView.canGoBack()) {
            webView.goBack()
        } else {
            super.onBackPressed()
        }
    }
}
```

5. Add internet permission in `AndroidManifest.xml`:

```xml
<uses-permission android:name="android.permission.INTERNET" />
```

6. Build APK:
   - Build
   - Build Bundle(s) / APK(s)
   - Build APK(s)

## Why this approach is acceptable academically

- The project logic remains fully your own Python code.
- The APK is only a container for the web app.
- No paid wrapper tools are needed.
- The same code works on Windows and Android.

## What to say in viva

If asked why you used this route:

1. Streamlit is best for rapid AI dashboard prototyping.
2. Native Android packaging is not Streamlit's core purpose.
3. A WebView or PWA path gives a free installable experience without rewriting the entire project in Flutter, Kotlin, or React Native.
4. This choice keeps the research contribution focused on AI, explainability, and adaptive lifestyle support instead of platform-specific UI engineering.

## If a fully native Android app is required later

Future upgrade options:

- Flutter front-end with FastAPI backend
- Kivy Python app
- Chaquopy hybrid Android app
- React Native with model API backend

For the current major project, the PWA/WebView route is the most realistic free deployment option.

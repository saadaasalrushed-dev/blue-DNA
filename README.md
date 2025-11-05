# Blue DNA - AI Beach Guardian

Clean, essential files only for deployment.

## 📁 Structure

```
blue_dna_clean/
├── app.py              # Main Flask application
├── model.py            # AI model loading
├── wsgi.py             # WSGI entry point
├── requirements.txt    # Python dependencies
├── Procfile            # Deployment config
├── templates/          # HTML templates
│   ├── index.html
│   ├── dashboard.html
│   ├── scanner.html
│   ├── map.html
│   └── info.html
├── static/
│   ├── css/
│   │   └── style.css   # Main stylesheet
│   ├── js/
│   │   └── app.js      # Main JavaScript
│   ├── images/
│   │   └── blue_dna_logo.svg
│   └── uploads/        # User uploads folder
└── models/
    └── pollution_classifier.h5  # Trained AI model
```

## 🚀 Deploy

1. Upload all files to GitHub
2. Connect to Render/Railway/etc
3. Deploy!

## ✅ Essential Files Only

- Removed all documentation (.md files)
- Removed extra CSS files (only style.css)
- Removed extra JS files (only app.js)
- Removed training scripts
- Removed development files


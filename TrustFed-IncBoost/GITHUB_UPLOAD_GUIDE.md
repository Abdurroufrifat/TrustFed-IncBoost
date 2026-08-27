# GitHub upload guide

## 1. Create the empty repository

Open <https://github.com/new> and use:

- Repository name: `TrustFed-IncBoost`
- Description: `Chance-aware trust-gated federated XGBoost for poisoning-resilient IoMT intrusion detection.`
- Visibility: **Public**
- Do not add another README, `.gitignore`, or license on GitHub.

Click **Create repository**.

## 2. Commit the prepared project

Open the VS Code PowerShell terminal in the project folder:

```powershell
cd D:\TrustFed-IncBoost
git init
git branch -M main
git add .
git status
git commit -m "Release TrustFed-IncBoost v1.0"
```

Before committing, confirm that `data/raw`, `.venv`, `outputs`, and `*.joblib` do not appear in the staged-file list.

## 3. Connect and push

```powershell
git remote add origin https://github.com/Abdurroufrifat/TrustFed-IncBoost.git
git push -u origin main
```

If Git reports that `origin` already exists, use:

```powershell
git remote set-url origin https://github.com/Abdurroufrifat/TrustFed-IncBoost.git
git push -u origin main
```

GitHub may open a browser window for authentication. Do not paste a password or token into a tracked file.

## 4. Create the frozen release tag

```powershell
git tag -a v1.0.0 -m "Frozen EHMS confirmation release"
git push origin v1.0.0
```

On GitHub, open **Releases**, choose **Draft a new release**, select `v1.0.0`, and title it:

```text
TrustFed-IncBoost v1.0 - Frozen EHMS Confirmation
```

## 5. Recommended repository settings

Add these topics:

```text
federated-learning intrusion-detection iomt cybersecurity xgboost data-poisoning non-iid
```

Never upload the raw WUSTL-EHMS CSV unless its provider explicitly permits redistribution.

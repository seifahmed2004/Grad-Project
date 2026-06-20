# Fix cv2 / libGL on Streamlit Cloud for Age + Gender

Run from:

```powershell
cd "C:\Users\lenovo\Downloads\final_4_models_integration_CODE_ONLY"
python .\fix_cv2_cloud_agegender_patch\apply_fix_cv2_cloud_agegender.py
```

Then:

```powershell
git add final_4_models_integration_CODE_ONLY/interfaces/age_gender_interface.py
git add final_4_models_integration_CODE_ONLY/requirements.txt
git add packages.txt
git add final_4_models_integration_CODE_ONLY/packages.txt
git commit -m "Make age gender interface cloud safe"
git push
```

Then on Streamlit Cloud:

Manage app -> Reboot. If old package cache persists, delete and deploy again.

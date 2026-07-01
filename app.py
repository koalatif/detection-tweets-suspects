# Point d'entree racine (Hugging Face Spaces / Streamlit Cloud).
import runpy, os
runpy.run_path(os.path.join("app", "streamlit_app.py"), run_name="__main__")

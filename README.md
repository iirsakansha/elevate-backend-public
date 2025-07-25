# elevate-backend set up

1. Create a Virtual Environment
A virtual environment helps isolate project dependencies.

python -m venv env

Creates a folder named env with an isolated Python environment.

2. Activate the Virtual Environment
On Windows:
.\env\Scripts\activate
On macOS/Linux:
source env/bin/activate

After activation, your terminal will be prefixed with (env), indicating you're inside the virtual environment.

3. Install Required Packages

 From requirements.txt

pip install -r requirements.txt
pip install django
pip install django-admin-interface
pip install djangorestframework
pip install django-cors-headers
pip install django-rest-knox
pip install seaborn
pip install altair
pip install scipy
pip install xlrd>=2.0.1
pip install openpyxl
pip install django-environ
pip install psycopg2-binary
pip install black isort flake8 pre-commit
pip install psycopg2-binary
pip install numpy pandas

These packages are used for admin UI, APIs, CORS handling, authentication, and data visualization.

4. Apply Migrations

python manage.py migrate

Applies all database migrations and sets up initial database tables.


5. Create Superuser

python manage.py createsuperuser

You'll be prompted for:
Username (e.g., admin)
Email address (e.g., admin@example.com)
Password
This account will be used to access the Django admin panel.


6. Run the Development Server

python manage.py runserver

backend :http://127.0.0.1:8000/
Admin Panel : http://127.0.0.1:8000/superadmin/



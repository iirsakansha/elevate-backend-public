# Elevate Backend Setup Guide

## Prerequisites

- Python 3.8 or higher
- PostgreSQL database
- Git

## Initial Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd elevate-backend
```

### 2. Create a Virtual Environment

A virtual environment helps isolate project dependencies.

```bash
python -m venv env
```

Creates a folder named `env` with an isolated Python environment.

### 3. Activate the Virtual Environment

**On Windows:**

```bash
.\env\Scripts\activate
```

**On macOS/Linux:**

```bash
source env/bin/activate
```

After activation, your terminal will be prefixed with `(env)`, indicating you're inside the virtual environment.

### 4. Install Required Packages

**From requirements.txt:**

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

**Or manually install the packages:**

```bash
pip install django djangorestframework django-rest-knox \
django-cors-headers django-admin-interface django-environ \
altair seaborn scipy numpy pandas matplotlib openpyxl xlrd>=2.0.1 \
psycopg2-binary

pip install black isort flake8 pre-commit
pip install python-dotenv
```

These packages are used for admin UI, APIs, CORS handling, authentication, and data visualization.

## Environment Configuration

### 5. Setup Environment Variables

**Copy the environment template:**

```bash
cp .env.template .env
```

**Fill in your environment-specific values in `.env`:**

- For **local development**, you can reference `.env.example` for sample values
- For **production deployment**, use appropriate production values

**Important:** Never commit actual `.env` files to the repository. Only the template should be versioned.

### 6. Configure Database

Make sure your PostgreSQL database is running and update the database settings in your `.env` file:

```bash
DB_NAME=your_database_name
DB_USER=your_database_user
DB_PASSWORD=your_database_password
DB_HOST=localhost  # or your database host
DB_PORT=5432
```

### 7. Apply Migrations

```bash
python manage.py migrate
```

Applies all database migrations and sets up initial database tables.

### 8. Create Superuser

```bash
python manage.py createsuperuser
```

You'll be prompted for:

- Username (e.g., admin)
- Email address (e.g., admin@example.com)
- Password

This account will be used to access the Django admin panel.

### 9. Run the Development Server

```bash
python manage.py runserver
```

**Access URLs:**

- Backend API: http://127.0.0.1:8000/
- Admin Panel: http://127.0.0.1:8000/superadmin/

## Team Setup Instructions

### For Team Members Pulling Latest Changes

When you pull the latest changes from the main branch:

1. **Activate your virtual environment:**

   ```bash
   # Windows
   .\env\Scripts\activate

   # macOS/Linux
   source env/bin/activate
   ```

2. **Install any new dependencies:**

   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

3. **Setup/Update environment file:**

   ```bash
   # If .env doesn't exist, copy from template
   cp .env.template .env

   # Fill in your local development values (ask team lead if needed)
   ```

4. **Run migrations (if any new migrations exist):**

   ```bash
   python manage.py migrate
   ```

5. **Start the development server:**
   ```bash
   python manage.py runserver
   ```

### Environment Files Explanation

- **`.env.template`** - Template file with all required variables (committed to Git)
- **`.env.example`** - Example values for local development (committed to Git)
- **`.env`** - Your actual environment file with real values (NOT committed to Git)
- **`.env.aws`** - Production values for AWS deployment (NOT committed to Git)

## Development Workflow

1. **Always work in a virtual environment**
2. **Copy `.env.template` to `.env` and fill in your values**
3. **Run migrations after pulling changes**
4. **Never commit `.env` files with real credentials**

## Common Issues & Solutions

### Virtual Environment Issues

- Make sure you see `(env)` in your terminal before running commands
- If packages are missing, run `pip install -r requirements.txt`

### Database Issues

- Ensure PostgreSQL is running
- Check database credentials in `.env`
- Run `python manage.py migrate` after database setup

### Environment Variable Issues

- Make sure `.env` file exists and is properly configured
- Check that all required variables are set (refer to `.env.template`)

## Additional Commands

### Code Formatting & Linting

```bash
# Format code
black .
isort .

# Check code quality
flake8
```

### Running Tests

```bash
python manage.py test
```

### Collecting Static Files (for production)

```bash
python manage.py collectstatic
```

## Support

If you encounter any issues during setup, please:

1. Check this README for common solutions
2. Ask team members or team lead
3. Create an issue in the repository with detailed error information

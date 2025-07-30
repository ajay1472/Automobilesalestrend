from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
import os
import math
import re

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-this'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///calculator_saas.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    subscription_plan = db.Column(db.String(20), default='free')
    calculations_used = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_reset = db.Column(db.DateTime, default=datetime.utcnow)

class Calculation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    expression = db.Column(db.String(500), nullable=False)
    result = db.Column(db.String(100), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# Subscription limits
PLAN_LIMITS = {
    'free': 10,
    'basic': 100,
    'premium': 1000
}

class Calculator:
    @staticmethod
    def safe_eval(expression):
        """Safely evaluate mathematical expressions"""
        # Remove spaces and validate characters
        expression = expression.replace(' ', '')
        
        # Only allow numbers, operators, parentheses, and basic math functions
        allowed_chars = re.compile(r'^[0-9+\-*/().sincotaqrtlgexp,\s]+$')
        if not allowed_chars.match(expression):
            raise ValueError("Invalid characters in expression")
        
        # Replace common math functions
        expression = expression.replace('sin', 'math.sin')
        expression = expression.replace('cos', 'math.cos')
        expression = expression.replace('tan', 'math.tan')
        expression = expression.replace('sqrt', 'math.sqrt')
        expression = expression.replace('log', 'math.log')
        expression = expression.replace('exp', 'math.exp')
        expression = expression.replace('pi', 'math.pi')
        expression = expression.replace('e', 'math.e')
        
        # Evaluate safely
        allowed_names = {
            "__builtins__": {},
            "math": math
        }
        
        try:
            result = eval(expression, allowed_names)
            return str(result)
        except Exception as e:
            raise ValueError(f"Calculation error: {str(e)}")

def reset_monthly_usage():
    """Reset usage counters for users if a month has passed"""
    users = User.query.all()
    for user in users:
        if user.last_reset < datetime.utcnow() - timedelta(days=30):
            user.calculations_used = 0
            user.last_reset = datetime.utcnow()
    db.session.commit()

def check_usage_limit(user):
    """Check if user has exceeded their plan limit"""
    reset_monthly_usage()
    limit = PLAN_LIMITS.get(user.subscription_plan, 10)
    return user.calculations_used < limit

@app.route('/')
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user = User.query.get(session['user_id'])
    recent_calculations = Calculation.query.filter_by(user_id=user.id)\
        .order_by(Calculation.timestamp.desc()).limit(10).all()
    
    return render_template('calculator.html', 
                         user=user, 
                         recent_calculations=recent_calculations,
                         plan_limits=PLAN_LIMITS)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        if User.query.filter_by(email=email).first():
            flash('Email already exists')
            return redirect(url_for('register'))
        
        user = User(
            email=email,
            password_hash=generate_password_hash(password)
        )
        db.session.add(user)
        db.session.commit()
        
        session['user_id'] = user.id
        flash('Registration successful!')
        return redirect(url_for('index'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        user = User.query.filter_by(email=email).first()
        
        if user and check_password_hash(user.password_hash, password):
            session['user_id'] = user.id
            return redirect(url_for('index'))
        else:
            flash('Invalid email or password')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('login'))

@app.route('/calculate', methods=['POST'])
def calculate():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    user = User.query.get(session['user_id'])
    
    if not check_usage_limit(user):
        return jsonify({
            'error': f'Monthly limit reached. Upgrade your plan to continue.',
            'limit_reached': True
        }), 429
    
    try:
        expression = request.json.get('expression', '')
        result = Calculator.safe_eval(expression)
        
        # Save calculation
        calculation = Calculation(
            user_id=user.id,
            expression=expression,
            result=result
        )
        db.session.add(calculation)
        
        # Update usage
        user.calculations_used += 1
        db.session.commit()
        
        return jsonify({
            'result': result,
            'calculations_used': user.calculations_used,
            'limit': PLAN_LIMITS[user.subscription_plan]
        })
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': 'Calculation failed'}), 500

@app.route('/pricing')
def pricing():
    return render_template('pricing.html', plan_limits=PLAN_LIMITS)

@app.route('/upgrade/<plan>')
def upgrade(plan):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    if plan not in PLAN_LIMITS:
        flash('Invalid plan')
        return redirect(url_for('pricing'))
    
    user = User.query.get(session['user_id'])
    user.subscription_plan = plan
    db.session.commit()
    
    flash(f'Successfully upgraded to {plan} plan!')
    return redirect(url_for('index'))

# HTML Templates
templates = {
    'base.html': '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}Calculator SaaS{% endblock %}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        .calculator-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 10px;
            max-width: 300px;
        }
        .btn-calc {
            height: 60px;
            font-size: 18px;
        }
        .usage-bar {
            height: 10px;
            background-color: #e9ecef;
            border-radius: 5px;
            overflow: hidden;
        }
        .usage-fill {
            height: 100%;
            background-color: #007bff;
            transition: width 0.3s ease;
        }
    </style>
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-primary">
        <div class="container">
            <a class="navbar-brand" href="{{ url_for('index') }}">Calculator SaaS</a>
            {% if session.user_id %}
            <div class="navbar-nav ms-auto">
                <a class="nav-link" href="{{ url_for('pricing') }}">Pricing</a>
                <a class="nav-link" href="{{ url_for('logout') }}">Logout</a>
            </div>
            {% endif %}
        </div>
    </nav>
    
    <div class="container mt-4">
        {% with messages = get_flashed_messages() %}
            {% if messages %}
                {% for message in messages %}
                    <div class="alert alert-info">{{ message }}</div>
                {% endfor %}
            {% endif %}
        {% endwith %}
        
        {% block content %}{% endblock %}
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    {% block scripts %}{% endblock %}
</body>
</html>
    ''',
    
    'calculator.html': '''
{% extends "base.html" %}

{% block content %}
<div class="row">
    <div class="col-md-6">
        <div class="card">
            <div class="card-header d-flex justify-content-between align-items-center">
                <h5 class="mb-0">Calculator</h5>
                <span class="badge bg-secondary">{{ user.subscription_plan.title() }} Plan</span>
            </div>
            <div class="card-body">
                <div class="mb-3">
                    <label class="form-label">Usage this month:</label>
                    <div class="usage-bar">
                        <div class="usage-fill" style="width: {{ (user.calculations_used / plan_limits[user.subscription_plan] * 100) }}%"></div>
                    </div>
                    <small class="text-muted">{{ user.calculations_used }} / {{ plan_limits[user.subscription_plan] }} calculations</small>
                </div>
                
                <div class="mb-3">
                    <input type="text" id="display" class="form-control form-control-lg" readonly>
                </div>
                
                <div class="calculator-grid mx-auto">
                    <button class="btn btn-secondary btn-calc" onclick="clearDisplay()">C</button>
                    <button class="btn btn-secondary btn-calc" onclick="deleteLast()">⌫</button>
                    <button class="btn btn-secondary btn-calc" onclick="appendToDisplay('/')">/</button>
                    <button class="btn btn-secondary btn-calc" onclick="appendToDisplay('*')">×</button>
                    
                    <button class="btn btn-outline-primary btn-calc" onclick="appendToDisplay('7')">7</button>
                    <button class="btn btn-outline-primary btn-calc" onclick="appendToDisplay('8')">8</button>
                    <button class="btn btn-outline-primary btn-calc" onclick="appendToDisplay('9')">9</button>
                    <button class="btn btn-secondary btn-calc" onclick="appendToDisplay('-')">-</button>
                    
                    <button class="btn btn-outline-primary btn-calc" onclick="appendToDisplay('4')">4</button>
                    <button class="btn btn-outline-primary btn-calc" onclick="appendToDisplay('5')">5</button>
                    <button class="btn btn-outline-primary btn-calc" onclick="appendToDisplay('6')">6</button>
                    <button class="btn btn-secondary btn-calc" onclick="appendToDisplay('+')">+</button>
                    
                    <button class="btn btn-outline-primary btn-calc" onclick="appendToDisplay('1')">1</button>
                    <button class="btn btn-outline-primary btn-calc" onclick="appendToDisplay('2')">2</button>
                    <button class="btn btn-outline-primary btn-calc" onclick="appendToDisplay('3')">3</button>
                    <button class="btn btn-success btn-calc" onclick="calculate()" rowspan="2">=</button>
                    
                    <button class="btn btn-outline-primary btn-calc" onclick="appendToDisplay('0')" style="grid-column: span 2;">0</button>
                    <button class="btn btn-outline-primary btn-calc" onclick="appendToDisplay('.')">.</button>
                </div>
                
                <div class="mt-3">
                    <div class="btn-group w-100" role="group">
                        <button class="btn btn-outline-info btn-sm" onclick="appendToDisplay('sin(')">sin</button>
                        <button class="btn btn-outline-info btn-sm" onclick="appendToDisplay('cos(')">cos</button>
                        <button class="btn btn-outline-info btn-sm" onclick="appendToDisplay('tan(')">tan</button>
                        <button class="btn btn-outline-info btn-sm" onclick="appendToDisplay('sqrt(')">√</button>
                        <button class="btn btn-outline-info btn-sm" onclick="appendToDisplay('log(')">log</button>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <div class="col-md-6">
        <div class="card">
            <div class="card-header">
                <h5 class="mb-0">Recent Calculations</h5>
            </div>
            <div class="card-body">
                {% if recent_calculations %}
                    <div class="list-group list-group-flush">
                        {% for calc in recent_calculations %}
                        <div class="list-group-item d-flex justify-content-between align-items-center">
                            <div>
                                <strong>{{ calc.expression }}</strong>
                                <br>
                                <span class="text-muted">= {{ calc.result }}</span>
                            </div>
                            <small class="text-muted">{{ calc.timestamp.strftime('%m/%d %H:%M') }}</small>
                        </div>
                        {% endfor %}
                    </div>
                {% else %}
                    <p class="text-muted">No calculations yet. Start calculating!</p>
                {% endif %}
            </div>
        </div>
    </div>
</div>
{% endblock %}

{% block scripts %}
<script>
function appendToDisplay(value) {
    document.getElementById('display').value += value;
}

function clearDisplay() {
    document.getElementById('display').value = '';
}

function deleteLast() {
    const display = document.getElementById('display');
    display.value = display.value.slice(0, -1);
}

async function calculate() {
    const expression = document.getElementById('display').value;
    if (!expression) return;
    
    try {
        const response = await fetch('/calculate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ expression: expression })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            document.getElementById('display').value = data.result;
            updateUsage(data.calculations_used, data.limit);
            setTimeout(() => location.reload(), 1000); // Refresh to show new calculation in history
        } else {
            if (data.limit_reached) {
                alert('Monthly calculation limit reached! Please upgrade your plan to continue.');
                window.location.href = '/pricing';
            } else {
                alert('Error: ' + data.error);
            }
        }
    } catch (error) {
        alert('Calculation failed. Please try again.');
    }
}

function updateUsage(used, limit) {
    const percentage = (used / limit) * 100;
    document.querySelector('.usage-fill').style.width = percentage + '%';
    document.querySelector('.text-muted').textContent = `${used} / ${limit} calculations`;
}

// Allow Enter key to calculate
document.getElementById('display').addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        calculate();
    }
});
</script>
{% endblock %}
    ''',
    
    'login.html': '''
{% extends "base.html" %}

{% block content %}
<div class="row justify-content-center">
    <div class="col-md-6">
        <div class="card">
            <div class="card-header">
                <h4 class="mb-0">Login</h4>
            </div>
            <div class="card-body">
                <form method="POST">
                    <div class="mb-3">
                        <label for="email" class="form-label">Email</label>
                        <input type="email" class="form-control" id="email" name="email" required>
                    </div>
                    <div class="mb-3">
                        <label for="password" class="form-label">Password</label>
                        <input type="password" class="form-control" id="password" name="password" required>
                    </div>
                    <button type="submit" class="btn btn-primary w-100">Login</button>
                </form>
                <div class="text-center mt-3">
                    <a href="{{ url_for('register') }}">Don't have an account? Register here</a>
                </div>
            </div>
        </div>
    </div>
</div>
{% endblock %}
    ''',
    
    'register.html': '''
{% extends "base.html" %}

{% block content %}
<div class="row justify-content-center">
    <div class="col-md-6">
        <div class="card">
            <div class="card-header">
                <h4 class="mb-0">Register</h4>
            </div>
            <div class="card-body">
                <form method="POST">
                    <div class="mb-3">
                        <label for="email" class="form-label">Email</label>
                        <input type="email" class="form-control" id="email" name="email" required>
                    </div>
                    <div class="mb-3">
                        <label for="password" class="form-label">Password</label>
                        <input type="password" class="form-control" id="password" name="password" required>
                    </div>
                    <button type="submit" class="btn btn-primary w-100">Register</button>
                </form>
                <div class="text-center mt-3">
                    <a href="{{ url_for('login') }}">Already have an account? Login here</a>
                </div>
            </div>
        </div>
    </div>
</div>
{% endblock %}
    ''',
    
    'pricing.html': '''
{% extends "base.html" %}

{% block content %}
<div class="row">
    <div class="col-12">
        <h2 class="text-center mb-5">Choose Your Plan</h2>
    </div>
</div>

<div class="row">
    <div class="col-md-4">
        <div class="card h-100">
            <div class="card-header text-center">
                <h4>Free</h4>
                <h2>$0<small class="text-muted">/month</small></h2>
            </div>
            <div class="card-body">
                <ul class="list-unstyled">
                    <li>✓ {{ plan_limits.free }} calculations per month</li>
                    <li>✓ Basic calculator functions</li>
                    <li>✓ Calculation history</li>
                </ul>
            </div>
            <div class="card-footer">
                <a href="{{ url_for('upgrade', plan='free') }}" class="btn btn-outline-primary w-100">Current Plan</a>
            </div>
        </div>
    </div>
    
    <div class="col-md-4">
        <div class="card h-100 border-primary">
            <div class="card-header text-center bg-primary text-white">
                <h4>Basic</h4>
                <h2>$9<small class="text-muted">/month</small></h2>
            </div>
            <div class="card-body">
                <ul class="list-unstyled">
                    <li>✓ {{ plan_limits.basic }} calculations per month</li>
                    <li>✓ All calculator functions</li>
                    <li>✓ Extended calculation history</li>
                    <li>✓ Priority support</li>
                </ul>
            </div>
            <div class="card-footer">
                <a href="{{ url_for('upgrade', plan='basic') }}" class="btn btn-primary w-100">Upgrade to Basic</a>
            </div>
        </div>
    </div>
    
    <div class="col-md-4">
        <div class="card h-100">
            <div class="card-header text-center">
                <h4>Premium</h4>
                <h2>$19<small class="text-muted">/month</small></h2>
            </div>
            <div class="card-body">
                <ul class="list-unstyled">
                    <li>✓ {{ plan_limits.premium }} calculations per month</li>
                    <li>✓ All calculator functions</li>
                    <li>✓ Unlimited calculation history</li>
                    <li>✓ Priority support</li>
                    <li>✓ API access</li>
                </ul>
            </div>
            <div class="card-footer">
                <a href="{{ url_for('upgrade', plan='premium') }}" class="btn btn-success w-100">Upgrade to Premium</a>
            </div>
        </div>
    </div>
</div>
{% endblock %}
    '''
}

# Create templates directory and files
import os
if not os.path.exists('templates'):
    os.makedirs('templates')

for filename, content in templates.items():
    with open(f'templates/{filename}', 'w') as f:
        f.write(content)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)

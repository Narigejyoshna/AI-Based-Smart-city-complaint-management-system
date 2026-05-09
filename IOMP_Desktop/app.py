from flask import Flask, render_template, request, redirect, jsonify, session
import sqlite3
import joblib
import uuid
from sklearn.metrics.pairwise import cosine_similarity
import math

app = Flask(__name__)
app.secret_key = "smartcity_secret_2026"  # needed for session

model = joblib.load("model/category_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

# ---------------- CREATE DATABASE ---------------- #
conn = sqlite3.connect("complaints.db")
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS complaints(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tracking_id TEXT,
    text TEXT,
    category TEXT,
    priority TEXT,
    consequences TEXT,
    latitude TEXT,
    longitude TEXT,
    status TEXT
)''')
conn.commit()
conn.close()


# ---------------- PRIORITY ---------------- #
def predict_priority(text):
    text = text.lower()
    if "school" in text or "accident" in text:
        return "CRITICAL"
    elif "hospital" in text or "overflow" in text:
        return "HIGH"
    else:
        return "LOW"


# ---------------- CONSEQUENCES ---------------- #
def predict_consequences(category, priority):
    mapping = {
        "Water": "Road damage, Water wastage, Property damage",
        "Sanitation": "Disease spread, Mosquito breeding, Health hazard",
        "Electricity": "Accident risk, Power outage, Business disruption",
        "Roads": "Traffic accidents, Vehicle damage, Commute delays"
    }
    consequence = mapping.get(category, "General public inconvenience")
    if priority == "CRITICAL":
        consequence += ", Immediate safety risk, Severe public impact"
    elif priority == "HIGH":
        consequence += ", Significant disruption expected"
    return consequence


# ---------------- HAVERSINE DISTANCE (km) ---------------- #
def haversine(lat1, lon1, lat2, lon2):
    try:
        R = 6371  # Earth radius in km
        dlat = math.radians(float(lat2) - float(lat1))
        dlon = math.radians(float(lon2) - float(lon1))
        a = math.sin(dlat/2)**2 + math.cos(math.radians(float(lat1))) * math.cos(math.radians(float(lat2))) * math.sin(dlon/2)**2
        return R * 2 * math.asin(math.sqrt(a))
    except:
        return 999  # If coords invalid, treat as far away


# ---------------- DUPLICATE CHECK (location-aware) ---------------- #
def is_duplicate(new_text, new_lat, new_lon):
    """
    A complaint is duplicate ONLY IF:
    1. Text similarity > 85%, AND
    2. Location is within 0.5 km (same area).
    Different locations = different complaints even if same problem.
    """
    conn = sqlite3.connect("complaints.db")
    c = conn.cursor()
    c.execute("SELECT text, latitude, longitude FROM complaints")
    old_rows = c.fetchall()
    conn.close()

    if not old_rows:
        return False

    old_texts = [r[0] for r in old_rows]
    texts = old_texts + [new_text]

    vectors = vectorizer.transform(texts)
    similarity_matrix = cosine_similarity(vectors)
    similarities = similarity_matrix[-1][:-1]

    for i, score in enumerate(similarities):
        if score > 0.85:
            old_lat = old_rows[i][1]
            old_lon = old_rows[i][2]

            # If either complaint has no location, fall back to text-only check
            if not old_lat or not old_lon or not new_lat or not new_lon:
                return True  # text match, no location to differentiate

            dist = haversine(new_lat, new_lon, old_lat, old_lon)
            if dist <= 0.5:  # Same place (within 500 meters) AND same problem = duplicate
                return True
            # else: same problem but different location → NOT duplicate

    return False


# ---------------- USER PORTAL ---------------- #
@app.route("/")
def user_portal():
    return render_template("user.html")


# ---------------- SUBMIT (with double-submit protection) ---------------- #
@app.route("/submit", methods=["POST"])
def submit():
    # --- Double-submit guard using session token ---
    form_token = request.form.get("form_token")
    if not form_token:
        return render_template("user.html", error="Invalid form submission.")

    if session.get("last_token") == form_token:
        # Same token = browser re-submitted (refresh / back button)
        last_id = session.get("last_tracking_id")
        return render_template("success.html", tracking_id=last_id, already_submitted=True)

    text = request.form["complaint"]
    latitude = request.form.get("latitude")
    longitude = request.form.get("longitude")

    # Location-aware duplicate check
    if is_duplicate(text, latitude, longitude):
        return render_template("duplicate.html")

    vec = vectorizer.transform([text])
    category = model.predict(vec)[0]
    priority = predict_priority(text)
    consequences = predict_consequences(category, priority)

    tracking_id = str(uuid.uuid4())[:8].upper()

    conn = sqlite3.connect("complaints.db")
    c = conn.cursor()
    c.execute("""
        INSERT INTO complaints
        (tracking_id, text, category, priority, consequences, latitude, longitude, status)
        VALUES (?,?,?,?,?,?,?,?)
    """, (tracking_id, text, category, priority, consequences, latitude, longitude, "Pending"))
    conn.commit()
    conn.close()

    # Store token so re-submit is caught
    session["last_token"] = form_token
    session["last_tracking_id"] = tracking_id

    return render_template("success.html", tracking_id=tracking_id, already_submitted=False)


# ---------------- ADMIN LOGIN PAGE ---------------- #
@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        # Simple hardcoded credentials — replace with DB check in production
        if username == "admin" and password == "12345":
            session["admin_logged_in"] = True
            return redirect("/admin")
        else:
            return render_template("admin_login.html", error="Invalid credentials")
    return render_template("admin_login.html", error=None)


@app.route("/admin/logout")
def admin_logout():
    session.pop("admin_logged_in", None)
    return redirect("/admin/login")


# ---------------- ADMIN DASHBOARD (protected) ---------------- #
@app.route("/admin")
def admin_dashboard():
    if not session.get("admin_logged_in"):
        return redirect("/admin/login")

    conn = sqlite3.connect("complaints.db")
    c = conn.cursor()
    c.execute("""
        SELECT * FROM complaints
        ORDER BY CASE priority
            WHEN 'CRITICAL' THEN 1
            WHEN 'HIGH' THEN 2
            ELSE 3
        END, id DESC
    """)
    data = c.fetchall()
    conn.close()
    return render_template("admin.html", data=data)


# ---------------- UPDATE STATUS ---------------- #
@app.route("/update_status/<tracking_id>/<new_status>")
def update_status(tracking_id, new_status):
    if not session.get("admin_logged_in"):
        return redirect("/admin/login")
    conn = sqlite3.connect("complaints.db")
    c = conn.cursor()
    c.execute("UPDATE complaints SET status=? WHERE tracking_id=?",
              (new_status, tracking_id))
    conn.commit()
    conn.close()
    return redirect("/admin")


# ---------------- API ENDPOINT FOR STATISTICS ---------------- #
@app.route("/api/stats")
def get_stats():
    conn = sqlite3.connect("complaints.db")
    c = conn.cursor()
    c.execute("SELECT priority, status FROM complaints")
    data = c.fetchall()
    conn.close()
    stats = {
        "total": len(data),
        "critical": len([d for d in data if d[0] == "CRITICAL"]),
        "high": len([d for d in data if d[0] == "HIGH"]),
        "low": len([d for d in data if d[0] == "LOW"]),
        "pending": len([d for d in data if d[1] == "Pending"]),
        "in_progress": len([d for d in data if d[1] == "In Progress"]),
        "resolved": len([d for d in data if d[1] == "Resolved"])
    }
    return jsonify(stats)


# ---------------- API ENDPOINT FOR CRITICAL ALERTS ---------------- #
@app.route("/api/critical-alerts")
def get_critical_alerts():
    conn = sqlite3.connect("complaints.db")
    c = conn.cursor()
    c.execute("SELECT tracking_id, text, consequences FROM complaints WHERE priority='CRITICAL' AND status!='Resolved'")
    alerts = c.fetchall()
    conn.close()
    return jsonify([{"tracking_id": a[0], "complaint": a[1], "consequences": a[2]} for a in alerts])


# ---------------- TRACK COMPLAINT ---------------- #
@app.route("/track", methods=["GET", "POST"])
def track():
    if request.method == "POST":
        tracking_id = request.form["tracking_id"]
        conn = sqlite3.connect("complaints.db")
        c = conn.cursor()
        c.execute("SELECT * FROM complaints WHERE tracking_id=?", (tracking_id,))
        complaint = c.fetchone()
        conn.close()
        return render_template("track.html", complaint=complaint)
    return render_template("track.html", complaint=None)


# ---------------- RUN ---------------- #
if __name__ == "__main__":
    app.run(debug=True)

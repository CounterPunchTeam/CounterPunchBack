from flask import request, jsonify
from app import app, db
from app.models import Fighter, Match, FighterScore
from datetime import datetime
from sqlalchemy import desc
import base64
import cv2
import numpy as np
import os
from roboflow import Roboflow

def decode_image(image_base64):
    image_data = base64.b64decode(image_base64.split(',')[1])
    np_arr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img

@app.route('/upload_frame', methods=['POST'])
def upload_frame():
    if 'frame' not in request.files:
        return jsonify({'error': 'No frame part'}), 400

    frame = request.files['frame']
    if frame.filename == '':
        return jsonify({'error': 'No selected frame'}), 400

    # Save the frame temporarily
    temp_filename = 'temp_frame.jpg'
    frame.save(temp_filename)

    # Initialize the Roboflow client
    rf = Roboflow(api_key="IhnzMIPA02sn9csVky59")
    project = rf.workspace().project("boxing-lelg6")
    model = project.version(3).model

    # Run inference
    prediction = model.predict(temp_filename, confidence=40, overlap=30).json()

    # Remove the temporary file
    os.remove(temp_filename)

    return jsonify(prediction), 200

@app.route('/fighter', methods=['POST'])
def create_fighter():
    data = request.json
    new_fighter = Fighter(name=data['name'], country=data['country'], avatarURL=data['avatarURL'])
    db.session.add(new_fighter)
    db.session.commit()
    return jsonify({"id": new_fighter.id}), 201

@app.route('/match', methods=['POST'])
def create_match():
    data = request.json
    new_match = Match(
        title=data['title'],
        datetime=datetime.fromisoformat(data['datetime']),
        fighter1_id=data['fighter1']['id'],
        fighter2_id=data['fighter2']['id']
    )
    new_match.fighter1_score = FighterScore(thrown=0, hits=0)
    new_match.fighter2_score = FighterScore(thrown=0, hits=0)
    db.session.add(new_match)
    db.session.commit()
    return jsonify({"id": new_match.id}), 201

@app.route('/match/<int:match_id>', methods=['GET'])
def get_match(match_id):
    match = Match.query.get_or_404(match_id)
    return jsonify({
        "id": match.id,
        "title": match.title,
        "datetime": match.datetime.isoformat(),
        "fighter1": {
            "id": match.fighter1.id,
            "name": match.fighter1.name,
            "country": match.fighter1.country,
            "avatarURL": match.fighter1.avatarURL
        },
        "fighter2": {
            "id": match.fighter2.id,
            "name": match.fighter2.name,
            "country": match.fighter2.country,
            "avatarURL": match.fighter2.avatarURL
        },
        "scores": {
            "fighter1": {
                "thrown": match.fighter1_score.thrown,
                "hits": match.fighter1_score.hits
            },
            "fighter2": {
                "thrown": match.fighter2_score.thrown,
                "hits": match.fighter2_score.hits
            }
        }
    })

@app.route('/match/<int:match_id>/score', methods=['PUT'])
def update_score(match_id):
    match = Match.query.get_or_404(match_id)
    data = request.json
    match.fighter1_score.thrown = data['scores']['fighter1']['thrown']
    match.fighter1_score.hits = data['scores']['fighter1']['hits']
    match.fighter2_score.thrown = data['scores']['fighter2']['thrown']
    match.fighter2_score.hits = data['scores']['fighter2']['hits']
    db.session.commit()
    return jsonify({"message": "Score updated successfully"}), 200

@app.route('/matches/recent', methods=['GET'])
def get_recent_matches():
    recent_matches = Match.query.order_by(desc(Match.datetime)).limit(5).all()
    
    matches_data = []
    for match in recent_matches:
        matches_data.append({
            "id": match.id,
            "title": match.title,
            "datetime": match.datetime.isoformat(),
            "fighter1": {
                "id": match.fighter1.id,
                "name": match.fighter1.name,
                "country": match.fighter1.country,
                "avatarURL": match.fighter1.avatarURL
            },
            "fighter2": {
                "id": match.fighter2.id,
                "name": match.fighter2.name,
                "country": match.fighter2.country,
                "avatarURL": match.fighter2.avatarURL
            },
            "scores": {
                "fighter1": {
                    "thrown": match.fighter1_score.thrown,
                    "hits": match.fighter1_score.hits
                },
                "fighter2": {
                    "thrown": match.fighter2_score.thrown,
                    "hits": match.fighter2_score.hits
                }
            }
        })
    
    return jsonify(matches_data), 200

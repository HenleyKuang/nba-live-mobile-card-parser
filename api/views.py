from django.http import HttpResponse
# from pymongo import MongoClient
from sklearn.externals import joblib
from nba_card_parser import parse_one, add_new_card_to_db, check_if_card_exist_in_db
from django.views.decorators.csrf import csrf_exempt
from pymongo import MongoClient
from PIL import Image
import os
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@csrf_exempt
def add_card(request):
    # import data into mongodb
    client = MongoClient('mongodb://henleyk:test1234@ds215388.mlab.com:15388/nbalivemobilecards')
    db = client.nbalivemobilecards
    cards_db = db.cards2
    card_dict = json.loads(request.body)
    # print card_dict
    response = {"status": add_new_card_to_db(cards_db, card_dict)}
    return HttpResponse(json.dumps(response))

@csrf_exempt
def exist_card(request):
    client = MongoClient('mongodb://henleyk:test1234@ds215388.mlab.com:15388/nbalivemobilecards')
    db = client.nbalivemobilecards
    cards_db = db.cards2
    card_hash = request.body
    response = {"status": check_if_card_exist_in_db(cards_db, card_hash)}
    return HttpResponse(json.dumps(response))

@csrf_exempt
def exist_card_list(request):
    client = MongoClient('mongodb://henleyk:test1234@ds215388.mlab.com:15388/nbalivemobilecards')
    db = client.nbalivemobilecards
    cards_db = db.cards2
    card_hash_list = json.loads(request.body)
    response = {}
    for index in card_hash_list:
        card_hash = card_hash_list[index]
        response[index] = check_if_card_exist_in_db(cards_db, card_hash)
    return HttpResponse(json.dumps(response))

@csrf_exempt
def parse_request(request):
    """
    List all code snippets, or create a new snippet.
    """
    # Load sk model
    adv_stats_clf, adv_stats_pp = joblib.load(os.path.join(BASE_DIR, 'Training/PKL/adv_stats_digits.pkl'))
    height_clf, height_pp = joblib.load(os.path.join(BASE_DIR, 'Training/PKL/height_digits.pkl'))
    ovr_clf, ovr_pp = joblib.load(os.path.join(BASE_DIR, 'Training/PKL/ovr_digits.pkl'))
    pos_clf, pos_pp = joblib.load(os.path.join(BASE_DIR, 'Training/PKL/pos.pkl'))
    type_clf, type_pp = joblib.load(os.path.join(BASE_DIR, 'Training/PKL/type.pkl'))
    # img_path = "../raw_cards\\271567.jpg"
    # img = Image.open(img_path)
    img = Image.open(request.FILES['file'])

    response = parse_one(img, adv_stats_clf, adv_stats_pp, height_clf, height_pp, ovr_clf, ovr_pp, pos_clf, pos_pp, type_clf, type_pp, True)
    return HttpResponse(json.dumps(response))

@csrf_exempt
def search(request):
    client = MongoClient('mongodb://henleyk:test1234@ds215388.mlab.com:15388/nbalivemobilecards')
    db = client.nbalivemobilecards
    cards_db = db.cards2
    cursor = cards_db.find({}, {'_id': False, 'card_img': False})
    response = list(cursor)
    return HttpResponse(json.dumps(response))

@csrf_exempt
def searchCardData(request):
    client = MongoClient('mongodb://henleyk:test1234@ds215388.mlab.com:15388/nbalivemobilecards')
    db = client.nbalivemobilecards
    cards_db = db.cards2
    card_hash = request.GET.get('hash')
    response = cards_db.find_one({'hash': card_hash}, {'_id': False, 'card_img': False})
    return HttpResponse(json.dumps(response))

@csrf_exempt
def searchCardImage(request):
    client = MongoClient('mongodb://henleyk:test1234@ds215388.mlab.com:15388/nbalivemobilecards')
    db = client.nbalivemobilecards
    cards_db = db.cards2
    card_hash = request.GET.get('hash')
    response = cards_db.find_one({'hash': card_hash}, {'_id': False, 'card_img': True})
    image_data = response['card_img']
    return HttpResponse(image_data.decode('base64'), content_type="image/png")

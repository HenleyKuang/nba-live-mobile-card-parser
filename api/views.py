from django.http import HttpResponse
# from pymongo import MongoClient
from sklearn.externals import joblib
from nba_card_parser import parse_one  # , add_new_card_to_db
from django.views.decorators.csrf import csrf_exempt
from PIL import Image
import json


@csrf_exempt
def add_card(request):
    # import data into mongodb
    # client = MongoClient('mongodb://henleyk:test1234@ds215388.mlab.com:15388/nbalivemobilecards')
    # db = client.nbalivemobilecards
    # cards_db = db.cards
    card_dict = request.body
    print card_dict
    response = card_dict  # add_new_card_to_db(cards_db, card_dict)
    return HttpResponse(json.dumps(response))


@csrf_exempt
def parse_request(request):
    """
    List all code snippets, or create a new snippet.
    """
    # Load sk model
    adv_stats_clf, adv_stats_pp = joblib.load('../Training/PKL/adv_stats_digits.pkl')
    height_clf, height_pp = joblib.load('../Training/PKL/height_digits.pkl')
    ovr_clf, ovr_pp = joblib.load('../Training/PKL/ovr_digits.pkl')
    pos_clf, pos_pp = joblib.load('../Training/PKL/pos.pkl')
    type_clf, type_pp = joblib.load('../Training/PKL/type.pkl')
    # img_path = "../raw_cards\\271567.jpg"
    # img = Image.open(img_path)
    img = Image.open(request.FILES['file'])
    response = parse_one(img, adv_stats_clf, adv_stats_pp, height_clf, height_pp, ovr_clf, ovr_pp, pos_clf, pos_pp, type_clf, type_pp)
    return HttpResponse(json.dumps(response))

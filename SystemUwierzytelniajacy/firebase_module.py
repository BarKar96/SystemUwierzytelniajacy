import pyrebase

config = {
    "apiKey": "AIzaSyCZC_LOaGfbJRniWbiNLB-Ues5vforzSHo",
    "authDomain": "personrecognizer-63e45.firebaseapp.com",
    "databaseURL": "https://personrecognizer-63e45.firebaseio.com",
    "projectId": "personrecognizer-63e45",
    "storageBucket": "",
    "messagingSenderId": "152064835675"
}

firebase = pyrebase.initialize_app(config)
auth = firebase.auth()



def createClient(email, password):

    try :
        user = auth.create_user_with_email_and_password(email,password)
    except Exception as e:
        print(e)

def signInClient(email, password):
    user = None
    try:
        user = auth.sign_in_with_email_and_password(email, password)
    except Exception as e:
        print(e)
    if user != None:
        print("udalo sie zalogowac")
        return 1
    else:
        print("nie udalo sie")
        return 0



#createClient()




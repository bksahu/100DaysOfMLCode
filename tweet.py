import tweepy

auth = tweepy.OAuthHandler('ONY6mw0VUYFnVZ9S72BZGQ2aY', 'AuxcQ2IOzgp2KT3NBwpfYzhPAm4IymXIsxXQ0j5MrierxHcBlx') #Fill these in
auth.set_access_token('878697948732825600-rq3plpgvcQuWHclsOXlm3N6odqqBVKd', 'JKTfzM4a7SmfASBZFDdvf9R9iSBKCmbd4MMZDlPRgd7WJ')  #Fill these in

api = tweepy.API(auth)



api.update_status(status ="Day 4 Naive Bayes Classifier of #100DaysOfMLCode Complete. https://github.com/bksahu/100DaysOfMLCode/tree/master/4.%20Naive%20Bayes%20Classifier")
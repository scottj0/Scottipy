#Author: John Scott
#GitHub: @scottj0
#This work is mine unless otherwise cited.

#This file contains the code used to sort existing playlists, by user defined criteria.
#The available conditions are taken from Spotify database end points publicly available.
#Graphs of the end points are also generated.

import spotipy
import spotipy.util as util
import random
from webbrowser import open_new_tab
from key import SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET #gets secret user keys from key.py
import base64 #this is needed to decode the playlist names from bytes to strings

import pandas as pd #Dataframe, Series
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns

import graphviz
import pydotplus
import io

import time

from scipy import misc

from sklearn.metrics import accuracy_score

a='cm9vdA=='
b=base64.b64decode(a).decode('utf-8')

red_blue = ['#19B5FE', '#EF4836']
palette = sns.color_palette(red_blue)
sns.set_palette(palette)
sns.set_style('white')

class User():
	def __init__(self):
		self.CLIENT_ID = SPOTIPY_CLIENT_ID
		self.CLIENT_SECRET = SPOTIPY_CLIENT_SECRET
		self.REDIRECT_URI = "http://localhost:5000"
		self.SCOPE = "playlist-read-private playlist-modify-private playlist-read-collaborative playlist-modify-public" #Allows program to access/edit the user's private and public playlists
		self.sp = self.getUser() #Creates Spotify instance
		self.id = self.sp.me()["id"] #Gets ID of authenticating user

	def getUser(self):
		#This function is required to authorize the application
		token = self.getUserToken()
		sp = spotipy.Spotify(auth=token)
		sp.trace = False
		return sp

	def getFeatures(self, track):
		#This function retrieves audio features from Spotify
		features = self.sp.audio_features(track)
		return features

	def getPlaylist(self):
		#This function gets all playlists from the user.
		results = self.sp.current_user_playlists()
		for i, item in enumerate(results["items"]):
			print ("{number} {name}".format(number=i, name=item["name"])) #Prints out the name of each playlist and a corresponding number

		choice = input("Please choose a playlist number: ")
		return results["items"][int(choice)]["id"]

	def getSongs(self, playlist_id):
		#This function gets the track IDs from the songs in the selected playlist.
		#It also generates graphs showing the features of the playlists' songs.
		results = self.sp.user_playlist_tracks(self.id,playlist_id)
		tracks = results["items"]
		song_ids = []
		while results["next"]:
			results = self.sp.next(results)
			tracks.extend(results["items"])
		for song in tracks:
			song_ids.append(song["track"]["id"])

		features = []
		j = 0
		for i in range(0,len(song_ids),50):
			audio_features = self.sp.audio_features(song_ids[i:i+50])
			for track in audio_features:
				features.append(track)
				track = tracks[j]
				j= j+1
				#features[-1]['trackPopularity'] = track['track']['popularity']
				#features[-1]['artistPopularity'] = self.sp.artist(track['track']['artists'][0]['id'])['popularity']
				features[-1]['target'] = 1
		j = 0

		trainingData = pd.DataFrame(features)
		trainingData.head()

		#train, test = train_test_split(trainingData, test_size = 0.15)
		#print("Training size: {}, Test size: {}".format(len(train),len(test)))

		red_blue = ['#19B5FE', '#EF4836']
		palette = sns.color_palette(red_blue)
		sns.set_palette(palette)
		sns.set_style('white')

		pos_dance = trainingData[trainingData['target'] == 1]['danceability']
		neg_dance = trainingData[trainingData['target'] == 0]['danceability']
		pos_energy = trainingData[trainingData['target'] == 1]['energy']
		neg_energy = trainingData[trainingData['target'] == 0]['energy']
		pos_loudness = trainingData[trainingData['target'] == 1]['loudness']
		neg_loudness = trainingData[trainingData['target'] == 0]['loudness']
		pos_acousticness = trainingData[trainingData['target'] == 1]['acousticness']
		neg_acousticness = trainingData[trainingData['target'] == 0]['acousticness']
		pos_instrumentalness = trainingData[trainingData['target'] == 1]['instrumentalness']
		neg_instrumentalness = trainingData[trainingData['target'] == 0]['instrumentalness']
		pos_liveness = trainingData[trainingData['target'] == 1]['liveness']
		neg_liveness = trainingData[trainingData['target'] == 0]['liveness']
		pos_valence = trainingData[trainingData['target'] == 1]['valence']
		neg_valence = trainingData[trainingData['target'] == 0]['valence']
		pos_tempo = trainingData[trainingData['target'] == 1]['tempo']
		neg_tempo = trainingData[trainingData['target'] == 0]['tempo']


		fig2 = plt.figure(figsize=(15,13))
		plt.subplots_adjust(hspace=0.42)
		#Danceability
		ax1 = fig2.add_subplot(331)
		ax1.set_xlabel('Danceability')
		ax1.set_ylabel('Count')
		ax1.set_title('Danceability Distribution')
		pos_dance.hist(alpha= 0.5, bins=30)
		ax2 = fig2.add_subplot(331)
		neg_dance.hist(alpha= 0.5, bins=30)

		#Energy
		ax3 = fig2.add_subplot(332)
		ax3.set_xlabel('Energy')
		ax3.set_ylabel('Count')
		ax3.set_title('Energy Distribution')
		pos_energy.hist(alpha= 0.5, bins=30)
		ax4 = fig2.add_subplot(332)
		neg_energy.hist(alpha= 0.5, bins=30)

		#Loudness
		ax5 = fig2.add_subplot(333)
		ax5.set_xlabel('Loudness')
		ax5.set_ylabel('Count')
		ax5.set_title('Loudness Distribution')
		pos_loudness.hist(alpha= 0.5, bins=30)
		ax6 = fig2.add_subplot(333)
		neg_loudness.hist(alpha= 0.5, bins=30)

		#Acousticness
		ax7 = fig2.add_subplot(334)
		ax7.set_xlabel('Acousticness')
		ax7.set_ylabel('Count')
		ax7.set_title('Acousticness Distribution')
		pos_valence.hist(alpha= 0.5, bins=30)
		ax8 = fig2.add_subplot(334)
		neg_valence.hist(alpha= 0.5, bins=30)

		#Instrumentalness
		ax9 = fig2.add_subplot(335)
		ax9.set_xlabel('Instrumentalness')
		ax9.set_ylabel('Count')
		ax9.set_title('Instrumentalness Distribution')
		pos_valence.hist(alpha= 0.5, bins=30)
		ax10 = fig2.add_subplot(335)
		neg_valence.hist(alpha= 0.5, bins=30)

		#Liveness
		ax11 = fig2.add_subplot(336)
		ax11.set_xlabel('Liveness')
		ax11.set_ylabel('Count')
		ax11.set_title('Liveness Distribution')
		pos_valence.hist(alpha= 0.5, bins=30)
		ax12 = fig2.add_subplot(336)
		neg_valence.hist(alpha= 0.5, bins=30)


		#Valence
		ax13 = fig2.add_subplot(337)
		ax13.set_xlabel('Valence')
		ax13.set_ylabel('Count')
		ax13.set_title('Song Valence Distribution')
		pos_valence.hist(alpha= 0.5, bins=30)
		ax14 = fig2.add_subplot(337)
		neg_valence.hist(alpha= 0.5, bins=30)

		#Tempo
		ax15 = fig2.add_subplot(338)
		ax15.set_xlabel('Tempo')
		ax15.set_ylabel('Count')
		ax15.set_title('Tempo Distribution')
		pos_valence.hist(alpha= 0.5, bins=30)
		ax16 = fig2.add_subplot(338)
		neg_valence.hist(alpha= 0.5, bins=30)

		plt.show()


		return song_ids


	def getUserToken(self):
		#This function is for user authentication
		name = "scottjohn0"
		token = util.prompt_for_user_token(username=name,scope=self.SCOPE, client_id=self.CLIENT_ID, client_secret=self.CLIENT_SECRET, redirect_uri=self.REDIRECT_URI)
		return token

	def sortSongs(self, songF, danceL, danceH, energyL, energyH, loudL, loudH, acousticL, acousticH,
		instrumentL, instrumentH, livenessL, livenessH, valenceL, valenceH, tempoL, tempoH):
		#This function returns true if the required song end points are met, adding 'true' to the list
		if danceL <= songF["danceability"] <= danceH:
			if energyL <= songF["energy"] <= energyH:
				if loudL <= songF["loudness"] <= loudH:
					if acousticL <= songF["acousticness"] <= acousticH:
						if instrumentL <= songF["instrumentalness"] <= instrumentH:
							if livenessL <= songF["liveness"] <= livenessH:
								if valenceL <= songF["valence"] <= valenceH:
									if tempoL <= songF["tempo"] <= tempoH:
										return True
	def getLimits(self):
		#This function allows the user to set limits on tracks. No response takes the lowest or highest in the range.
		danceL = float(input("Danceability minimum (how suitable track is for dancing 0.0-1.0): ") or "0")
		danceH = float(input("Danceability maximum: ") or "1")
		energyL = float(input("Energy minimum (intensity, or speed of a track 0.0-1.0): ") or "0")
		energyH = float(input("Energy maximum: ") or "1")
		loudL = float(input("Loudness minimum (Overall loudness of a track in decibels -60-0): ") or "-60")
		loudH = float(input("Loudness maximum: ") or "0")
		acousticL = float(input("Acousticness minimum (measure of whether a track is acoustic 0.0-1): ") or "0")
		acousticH = float(input("Acousticness maximum: ") or "1")
		instrumentL = float(input("Instrumentalness minimum (Predicts whether track contains no vocals 0.0-1.0): ") or "0")
		instrumentH = float(input("Instrumentalness maximum: ") or "1")
		livenessL = float(input("Liveness minimum (Detects presence of audience 0.0-1.0): ") or "0")
		livenessH = float(input("Liveness maximum: ") or "1")
		valenceL = float(input("Valence minimum (Positivity measurement 0.0-1.0): ") or "0")
		valenceH = float(input("Valence maximum: ") or "1")
		tempoL = float(input("Tempo minimum: ") or "0")
		tempoH = float(input("Tempo maximum: ") or "300")
		name = input("Please name your playlist: ") #allows user to name the new playlist
		return [danceL, danceH, energyL, energyH, loudL, loudH, acousticL, acousticH, instrumentL, instrumentH, livenessL, livenessH, valenceL, valenceH, tempoL, tempoH, name]

	def createPlaylist(self, title, tracks):
		#Makes the playlist
		playlist = self.sp.user_playlist_create(self.id, title, False)
		for track in tracks:
			self.sp.user_playlist_add_tracks(self.id, playlist['id'], [track])

		features = []
		j = 0
		for i in range(0,len(tracks),50):
			new_tracks = self.sp.audio_features(tracks[i:i+50])
			for track in new_tracks:
				features.append(track)
				track = tracks[j]
				j= j+1
				features[-1]['target'] = 1
		j = 0

		trainingData = pd.DataFrame(features)
		trainingData.head()

		red_blue = ['#19B5FE', '#EF4836']
		palette = sns.color_palette(red_blue)
		sns.set_palette(palette)
		sns.set_style('white')

		pos_dance = trainingData[trainingData['target'] == 1]['danceability']
		neg_dance = trainingData[trainingData['target'] == 0]['danceability']
		pos_energy = trainingData[trainingData['target'] == 1]['energy']
		neg_energy = trainingData[trainingData['target'] == 0]['energy']
		pos_loudness = trainingData[trainingData['target'] == 1]['loudness']
		neg_loudness = trainingData[trainingData['target'] == 0]['loudness']
		pos_acousticness = trainingData[trainingData['target'] == 1]['acousticness']
		neg_acousticness = trainingData[trainingData['target'] == 0]['acousticness']
		pos_instrumentalness = trainingData[trainingData['target'] == 1]['instrumentalness']
		neg_instrumentalness = trainingData[trainingData['target'] == 0]['instrumentalness']
		pos_liveness = trainingData[trainingData['target'] == 1]['liveness']
		neg_liveness = trainingData[trainingData['target'] == 0]['liveness']
		pos_valence = trainingData[trainingData['target'] == 1]['valence']
		neg_valence = trainingData[trainingData['target'] == 0]['valence']
		pos_tempo = trainingData[trainingData['target'] == 1]['tempo']
		neg_tempo = trainingData[trainingData['target'] == 0]['tempo']


		fig = plt.figure(figsize=(15,15))
		plt.subplots_adjust(hspace=0.42)
		#Danceability
		ax1 = fig.add_subplot(331)
		ax1.set_xlabel('Danceability')
		ax1.set_ylabel('Count')
		ax1.set_title('Danceability Distribution')
		pos_dance.hist(alpha= 0.5, bins=30)
		ax2 = fig.add_subplot(331)
		neg_dance.hist(alpha= 0.5, bins=30)

		#Energy
		ax3 = fig.add_subplot(332)
		ax3.set_xlabel('Energy')
		ax3.set_ylabel('Count')
		ax3.set_title('Energy Distribution')
		pos_energy.hist(alpha= 0.5, bins=30)
		ax4 = fig.add_subplot(332)
		neg_energy.hist(alpha= 0.5, bins=30)

		#Loudness
		ax5 = fig.add_subplot(333)
		ax5.set_xlabel('Loudness')
		ax5.set_ylabel('Count')
		ax5.set_title('Loudness Distribution')
		pos_loudness.hist(alpha= 0.5, bins=30)
		ax6 = fig.add_subplot(333)
		neg_loudness.hist(alpha= 0.5, bins=30)

		#Acousticness
		ax7 = fig.add_subplot(334)
		ax7.set_xlabel('Acousticness')
		ax7.set_ylabel('Count')
		ax7.set_title('Acousticness Distribution')
		pos_valence.hist(alpha= 0.5, bins=30)
		ax8 = fig.add_subplot(334)
		neg_valence.hist(alpha= 0.5, bins=30)

		#Instrumentalness
		ax9 = fig.add_subplot(335)
		ax9.set_xlabel('Instrumentalness')
		ax9.set_ylabel('Count')
		ax9.set_title('Instrumentalness Distribution')
		pos_valence.hist(alpha= 0.5, bins=30)
		ax10 = fig.add_subplot(335)
		neg_valence.hist(alpha= 0.5, bins=30)

		#Liveness
		ax11 = fig.add_subplot(336)
		ax11.set_xlabel('Liveness')
		ax11.set_ylabel('Count')
		ax11.set_title('Liveness Distribution')
		pos_valence.hist(alpha= 0.5, bins=30)
		ax12 = fig.add_subplot(336)
		neg_valence.hist(alpha= 0.5, bins=30)


		#Valence
		ax13 = fig.add_subplot(337)
		ax13.set_xlabel('Valence')
		ax13.set_ylabel('Count')
		ax13.set_title('Song Valence Distribution')
		pos_valence.hist(alpha= 0.5, bins=30)
		ax14 = fig.add_subplot(337)
		neg_valence.hist(alpha= 0.5, bins=30)

		#Tempo
		ax15 = fig.add_subplot(338)
		ax15.set_xlabel('Tempo')
		ax15.set_ylabel('Count')
		ax15.set_title('Tempo Distribution')
		pos_valence.hist(alpha= 0.5, bins=30)
		ax16 = fig.add_subplot(338)
		neg_valence.hist(alpha= 0.5, bins=30)

		plt.show()

		open_new_tab("https://open.spotify.com/collection/playlists") #uses web browser to view playlist

	def main(self):
		playlist = self.getPlaylist()
		songs = self.getSongs(playlist)
		newPlaylist = []
		pref = self.getLimits()
		for song_id in songs:
			song = self.getFeatures([song_id])
			if self.sortSongs(song[0], pref[0], pref[1], pref[2], pref[3], pref[4], pref[5], pref[6], pref[7], pref[8], pref[9], pref[10], pref[11], pref[12], pref[13], pref[14], pref[15]):
				newPlaylist.append(song[0]['id'])

		self.createPlaylist(pref[16], newPlaylist)

if __name__ == "__main__":
	SpotifyUser = User()
	SpotifyUser.main()

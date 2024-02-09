import PySimpleGUI as sg

class GUI:

	def __init__(self) -> None:
		pass

	def Directory_photo(self):
		dir_ph=""
		sg.theme("DarkTeal2")
		layout = [[sg.T("")], [sg.Text("Choose a file: "), sg.Input(), sg.FileBrowse(key="-IN-")], [sg.Button("Submit")]]

		###Building Window
		window = sg.Window('Choose photo', layout, size=(600, 150))

		while True:
			event, values = window.read()
			if event == sg.WIN_CLOSED or event == "Exit":
				break
			elif event == "Submit":
				dir_ph=values["-IN-"]
				break
		window.close()
		return dir_ph
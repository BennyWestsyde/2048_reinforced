from dataclasses import dataclass
@dataclass(frozen=True)
class Colors:
	fg = {
		'black': '30',
		'red': '31',
		'green': '32',
		'yellow': '33',
		'blue': '34',
		'magenta': '35',
		'cyan': '36',
		'white': '37',
		}
	bg = {
		'black': '40',
		'red': '41',
		'green': '42',
		'yellow': '43',
		'blue': '44',
		'magenta': '45',
		'cyan': '46',
		'white': '47',
		}
	styles = {
		'reset': '0',
		'bold': '1',
		'underline': '4',
		'blink': '5',
		'reverse': '7',
		'conceal': '8',
		}
	color_list = list(fg.keys()) + list(bg.keys())
	style_list = list(styles.keys())

def color(text, fg=None, bg=None, style=None):
	if fg not in Colors.color_list:
		fg = None
	if bg not in Colors.color_list:
		bg = None
	if style not in Colors.style_list:
		style = None
	color_code = []
	if fg:
		color_code.append(Colors.fg[fg])
	if bg:
		color_code.append(Colors.bg[bg])
	if style:
		color_code.append(Colors.styles[style])
	if color_code:
		return f'\033[{";".join(color_code)}m{text}\033[0m'
	else:
		return text

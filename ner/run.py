# %%
import time
import torch
from gliner import GLiNER
import json

model = GLiNER.from_pretrained("knowledgator/gliner-multitask-large-v0.5", local_files_only=True)

text = """
Auch die Lage im bayerischen Regensburg in der Oberpfalz ist angespannt. "Wir haben noch ein, zwei Tage echte Anspannung", sagte Oberbürgermeisterin Gertrud Maltz-Schwarzfischer bei einem Besuch von Bayerns Ministerpräsident Markus Söder. Sie hoffe, dass die bisher aufgebauten Schutzmaßnahmen auf den Donauinseln standhalten. Der Wasserstand der Donau halte sich derzeit auf hohem Niveau, sinke bislang aber nicht.
Bundeswirtschaftsminister Robert Habeck sagte den von den Fluten betroffenen Menschen verlässliche Unterstützung zu. "In den Hochwassergebieten steht jetzt nur eins im Vordergrund: Leib und Leben zu retten. Das ist der Imperativ der Stunde. Den Menschen in den Überschwemmungsgebieten muss aber auch beim Wiederaufbau geholfen werden", sagte er der Augsburger Allgemeinen.Die bayerische Staatsregierung will mindestens 100 Millionen Euro an Finanzhilfen für Betroffene bereitstellen: "100 Millionen plus X", sagte Ministerpräsident Markus Söder nach einem entsprechenden Kabinettsbeschluss in München. Von dem Hilfspaket sollen grundsätzlich sowohl Privathaushalte als auch Gewerbebetriebe, Selbstständige sowie Land- und Forstwirte profitieren können. "Bayern hilft, schnell und unbürokratisch", sagte Söder. Der CSU-Politiker forderte aber auch den Bund auf, seine Zusagen einzuhalten und Fluthilfe zu leisten.Auch SPD-Fraktionschef Rolf Mützenich drängte auf staatliche Hilfe für die Betroffenen. "Menschen müssen sich darauf verlassen, in solchen elementaren Situationen auch die Hilfe des Staates in Anspruch nehmen zu können - und das wollen wir auch gewährleisten", sagte er vor einer Fraktionssitzung in Berlin. Er hoffe, dass dies in den laufenden Haushaltsverhandlungen berücksichtigt werde. "Ich finde, ein starker Staat wie Deutschland kann es sich leisten, auch letztlich eben diese Hilfszusagen zu machen", betonte Mützenich.
"""

labels = ["person name"]
# %%
import pyinstrument as pyi
torch.set_num_threads(8)
model.eval()
with pyi.Profiler() as p:
    with torch.no_grad():
        tic = time.perf_counter()
        entities = model.predict_entities(text, labels=labels, threshold=0.2)
        tac = time.perf_counter()
        print(tac - tic)
p.print(show_all=True, color=True)

print(json.dumps(entities, indent=2))
for entity in entities:
    print(entity["text"], "=>", entity["label"])

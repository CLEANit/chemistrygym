# chemistrygym
A  chemistry laboratory environment populated with a collection of chemistry experiment sub-environments, based on the OpenAI Gym environment framework for use with reinforcement learning applications.
## Data Structure
volume: ml, amount: mole
### Vessel
Important Variables |Structure | Description
---|---|---
_material_dict|{'material.name': [material(), amount], ...}|a dictionary holding all the material inside this vessel
_solute_dict|{'solute.name': {solvent.name: amount, ...}, ...}|dictionary that represents the solution
_event_dict|{'function name': function}|a dictionary holds the event functions of a vessel
_event_queue|[['event', parameters], ['event', parameters] ... ]|a queue of events to be performed by vessel
_feedback_queue|[['event', parameters], ['event', parameters] ... ]|a queue holding collected feedback from materials and unfinished events

#### An example of _material_dict and _solute_dict
(vessel with Na and Cl dissolved in H2O and C6H14)
```bash
H2O = material.H2O()
C6H14 = material.C6H14()
Na = material.Na()
Cl = material.Cl()

Na.set_charge(1.0)
Na.set_solute_flag(True)
Cl.set_charge(-1.0)
Cl.set_solute_flag(True)

material_dict = {'H2O': [H2O, 100], 'C6H14': [C6H14, 100], 'Na': [Na, 1.0], 'Cl': [Cl, 1.0]}
solute_dict = {'Na': {'H2O': 0.5, 'C6H14': 0.5}, 'Cl': {'H2O': 0.5, 'C6H14': 0.5}}
```

Important functions | Description
---|---
push_event_to_queue()|used to pass event into the vessel
_update_materials()|automatically called by push_event_to_queue(), execute events in _event_queue and _feedback_queue
_merge_event_queue()|merge the feedback_queue passed in, and all the default events (switches) are appended to _feedback_queue by this function

#### The Workflow
  
  1. Agent choose action from the action space of an environment.
  2. The environment does the calculation and update and generate events.
  3. At the end of each action, if the action affect a vessel, use push_event_to_queue() to push the event into the vessel, if no event generated, call the function with events=None.
  4. With push_event_to_queue() called, events are pushed into the vessel.
  5. _update_materials is automatically called and perform events in the events_queue.
  6. Each event has a corresponding event function, it first update properties of the vessel, then loop over the materials inside the vessel by calling the corresponding event functions of each material.
  7. The materials' event function will return feedback by calling the push_event_to_queue(), which contains feedback and unfinished event 
  8. The returned feedback is added to the _feedback_queue
  9. The the _merge_event_queue() is called on _feedback_queue, which merge the events in the feedback_queue to generate a merged_queue and add default event into it, then empty the _feedback_queue
  10. Then the merged_queue will be executed and new feedback are collected and added to _feedback_queue, which will be executed with the next action. 

#### Event Functions
Function Name|Description
---|---
'pour by volume'|Pour from self vessel to target vessel by certain volume
'drain by pixel|Drain from self vessel to target vessel by certain pixel
'fully mix'|Shake self vessel to fully mix
'update material dict'|Use input to update self vessel's material_dict
'update solute dict'|Use input to update self vessel's solute_dict
'mix'|Shake vessel of let vessel settle
'update_layer'|Update self vessel's layer representation

### Material

Properties|Description
---|---
name|Name of the material
density|Density of the material, g/mL
polarity|Polarity of the material
temperature|Temperature of the material, K
pressure|Pressure of the material, kPa
phase|Phase of the material (g, l, s)
charge|Charge of the material
molar_mass|Molar mass of the material, g/mol
color|A unique float number between 0 to 1 to represent the color of the the material, used in layer representation
solute|A flag showing if it's the solute in vessel (True/False)
solvent|A flag showing if it's the solvent in vessel (True/False)
index|A integer number, unique for each material, used to fix the location of the material in state (matrix)
 

### Util Functions
Function|Description
---|---
convert_material_dict_to_volume|Convert the input material_dict to a dictionary of {material's name: volume}
convert_volume_to_mole|Handy function to calculate mole
organize_material_dict|Organize the input material_dict, called by update_material_dict()
organize_solute_dict|Organize the input solute_dict, called by update_solute_dict()
check_overflow|Check if there's overflow, and calculate what remains in the vessel (always call it before push new material_dict and solute_dict to target vessel)
generate_state|generate the state representation from a list of vessels

### State Structure
Each vessel has three matrices to represent the state:

[material_dict_matrix, solute_dict_matrix, layer_vector]

**material_dict_matrix (number of available material * 10):**

-|Density|Polarity|Temperature|Pressure|Solid Flag|Liquid Flag|Gas Flag|Charge|Molar Mass|Amount
---|---|---|---|---|---|---|---|---|---|---
Air|
H2O|
H|
H2|
.|
.|
.|

**solute_dict_matrix (number of available material * number of available material):**

Each cell means how much of that solute (row) dissolved in that solvent (column)

-|Air|H2O|H|H2|.|.|.
---|---|---|---|---|---|---|----
Air|
H2O|
H|
H2|
.|
.|
.|

**layer_vector (1 * number of pixel):**

acquired form vessel.get_layers()

## To install:
```bash
pip install ./
```
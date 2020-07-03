'''
Module to describe the extraction of a desired material generated in a Wurtz reaction.

:title: wurtz_v0.py

:author: Mitchell Shahen

:history: 2020-06-27
'''

import numpy as np
import gym
import math
import copy
import sys

sys.path.append("../../../") # to access chemistrylab
from chemistrylab.chem_algorithms import material, util, vessel
from chemistrylab.extract_algorithms import separate
from chemistrylab.reactions.get_reactions import convert_to_class

class Extraction:
    '''
    Class object for a Wurtz extraction experiment
    '''

    def __init__(
        self,
        extraction_vessel,
        solute,
        target_material,
        n_empty_vessels=2,  # number of empty vessels
        solute_volume=1000000000,  # the amount of solute available (unlimited)
        dt=0.05,  # time for each step (time for separation)
        max_vessel_volume=1000.0,  # max volume of empty vessels / g
        n_vessel_pixels=100,  # number of pixels for each vessel
        max_valve_speed=10,  # maximum draining speed (pixels/step)
        n_actions=8
    ):
        '''
        Constructor class for the Wurtz Extraction class
        '''

        # set self variable
        self.dt = dt
        self.n_actions = n_actions
        self.max_vessel_volume = max_vessel_volume
        self.solute_volume = solute_volume
        self.n_empty_vessels = n_empty_vessels
        self.n_total_vessels = n_empty_vessels + 1  # (empty + input)
        self.n_vessel_pixels = n_vessel_pixels
        self.max_valve_speed = max_valve_speed

        self.solute = solute
        self.target_material = target_material
        self.target_material_init_amount = extraction_vessel.get_material_amount(target_material)

    # def get_observation_space(self):
    #     obs_low = np.zeros((self.n_total_vessels, self.n_vessel_pixels), dtype=np.float32)
    #     obs_high = 1.0 * np.ones((self.n_total_vessels, self.n_vessel_pixels), dtype=np.float32)
    #     observation_space = gym.spaces.Box(obs_low, obs_high, dtype=np.float32)
    #     return observation_space

    def get_action_space(self):
        """
        0: Valve (Speed multiplier, relative to max_valve_speed)
        1: Mix ExV (mixing coefficient, *-1 when passed into mix function)
        2: Pour B1 into ExV (Volume multiplier, relative to max_vessel_volume)
        3: Pour B2 into ExV (Volume multiplier, relative to max_vessel_volume)
        4: Pour ExV into B2 (Volume multiplier, relative to max_vessel_volume)
        5: Pour S1 into ExV (Volume multiplier, relative to max_vessel_volume)
        6: Pour S2 into ExV (Volume multiplier, relative to max_vessel_volume)
        7: Done (Value doesn't matter)

        :return: action_space
        """

        # action_space = gym.spaces.Box(low=np.array([0, 0], dtype=np.float32),
        #                               high=np.array([self.n_actions, 1], dtype=np.float32),
        #                               dtype=np.float32)

        action_space = gym.spaces.MultiDiscrete([self.n_actions, 5])

        return action_space

    def reset(self, extraction_vessel):
        '''
        Method to reset the environment
        '''

        # delete the extraction vessel's solute_dict and copy it into a list of vessels
        solute_dict = extraction_vessel._solute_dict
        extraction_vessel._solute_dict = {}
        vessels = [copy.deepcopy(extraction_vessel)]

        # create all the necessary beakers and add them to the list
        for i in range(self.n_empty_vessels):
            temp_vessel = vessel.Vessel(
                label='beaker_{}'.format(i + 1),
                v_max=self.max_vessel_volume,
                default_dt=0.05,
                n_pixels=self.n_vessel_pixels
            )
            vessels.append(temp_vessel)

        # generate a list of external vessels to contain solutes
        external_vessels = []

        # generate a vessel to contain the main solute
        solute_vessel = vessel.Vessel(
            label='solute_vessel0',
            v_max=self.solute_volume,
            n_pixels=self.n_vessel_pixels,
            settling_switch=False,
            layer_switch=False,
        )

        # create the material dictionary for the solute vessel
        solute_material_dict = {}
        solute_class = convert_to_class(materials=[self.solute])[0]
        solute_material_dict[self.solute] = [solute_class, self.solute_volume]

        # check for overflow
        solute_material_dict, _, _ = util.check_overflow(
            material_dict=solute_material_dict,
            solute_dict={},
            v_max=solute_vessel.get_max_volume()
        )

        # instruct the vessel to update its material dictionary
        event = ['update material dict', solute_material_dict]
        solute_vessel.push_event_to_queue(feedback=[event], dt=0)

        # add the main solute vessel to the list of external vessels
        external_vessels.append(solute_vessel)

        # generate vessels for each solute in the extraction vessel
        for solute_name in solute_dict:
            # generate an empty vessel to be filled with a single solute
            solute_vessel = vessel.Vessel(
                label='solute_vessel{}'.format(len(external_vessels)),
                v_max=extraction_vessel.v_max,
                n_pixels=self.n_vessel_pixels,
                settling_switch=False,
                layer_switch=False
            )
            solute_material_dict = {}
            solute_material_dict[solute_name] = solute_dict[solute_name]

            # check for overflow
            solute_material_dict, _, _ = util.check_overflow(
                material_dict=solute_material_dict,
                solute_dict={},
                v_max=solute_vessel.get_max_volume()
            )

            # instruct the vessel to update its material dictionary
            event = ['update material dict', solute_material_dict]
            solute_vessel.push_event_to_queue(feedback=[event], dt=0)

            # add this solute vessel to the list of external vessels
            external_vessels.append(solute_vessel)

        # generate the state
        state = util.generate_state(
            vessel_list=vessels,
            max_n_vessel=self.n_total_vessels
        )

        return vessels, external_vessels, state

    def perform_action(self, vessels, external_vessels, action):
        """
        0: Valve (Speed multiplier, relative to max_valve_speed)
        1: Mix ExV (mixing coefficient, *-1 when passed into mix function)
        2: Pour B1 into ExV (Volume multiplier, relative to max_vessel_volume)
        3: Pour B2 into ExV (Volume multiplier, relative to max_vessel_volume)
        4: Pour ExV into B2 (Volume multiplier, relative to max_vessel_volume)
        5: Pour S1 into ExV (Volume multiplier, relative to max_vessel_volume)
        6: Pour S2 into ExV (Volume multiplier, relative to max_vessel_volume)
        7: Done (Value doesn't matter)

        :param external_vessels:
        :param vessels: a list containing several vessels
        :param action: a tuple of two elements, first one represents action, second one is a multiplier

        :return: new vessels, oil vessel, reward, done
        """

        reward = -1  # default reward for each step
        done = False
        do_action = int(action[0])
        multiplier = (action[1]) / 4 if action[1] != 0 else 0

        # if the multiplier is 0, push an empty list of events to the vessel queue
        if all([multiplier == 0, do_action != 7]):
            for vessel in vessels:
                __ = vessel.push_event_to_queue(dt=self.dt)
        else:
            # obtain the necessary vessels
            extract_vessel = vessels[0]
            beaker_1 = vessels[1]
            beaker_2 = vessels[2]
            solute_vessel1 = external_vessels[0]
            solute_vessel2 = external_vessels[1]

            # Open Valve (Speed multiplier)
            if do_action == 0:
                # calculate the number of pixels being drained
                drained_pixels = multiplier * self.max_valve_speed

                # drain the extraction vessel into the first beaker;
                event = ['drain by pixel', beaker_1, drained_pixels]

                # push the event to the extraction vessel
                reward = extract_vessel.push_event_to_queue(events=[event], dt=self.dt)

                # push no events to the second beaker
                __ = beaker_2.push_event_to_queue(dt=self.dt)

            # Mix the Extraction Vessel
            if do_action == 1:
                # mix the extraction vessel
                event = ['mix', -multiplier]

                # push the event to the extraction vessel
                reward = extract_vessel.push_event_to_queue(events=[event], dt=self.dt)

                # push no events to either beaker
                __ = beaker_1.push_event_to_queue(dt=self.dt)
                __ = beaker_2.push_event_to_queue(dt=self.dt)

            # pour Beaker 1 into the Extraction Vessel
            if do_action == 2:
                # determine the volume to pour from the first beaker into the extraction vessel
                d_volume = beaker_1.get_max_volume() * multiplier

                # push the event to the first beaker
                event = ['pour by volume', extract_vessel, d_volume]
                reward = beaker_1.push_event_to_queue(events=[event], dt=self.dt)

                # push no events to the second beaker
                __ = beaker_2.push_event_to_queue(dt=self.dt)

            # Pour Beaker 2 into the Extraction Vessel
            if do_action == 3:
                # determine the volume to pour from the second beaker into the extraction vessel
                d_volume = beaker_2.get_max_volume() * multiplier

                # push the event to the second beaker
                event = ['pour by volume', extract_vessel, d_volume]
                reward = beaker_2.push_event_to_queue(events=[event], dt=self.dt)

                # push no events to the first beaker
                beaker_1.push_event_to_queue(dt=self.dt)

            # Pour the Extraction Vessel into Beaker 2
            if do_action == 4:
                # determine the volume to pour from the extraction vessel into the second beaker
                d_volume = extract_vessel.get_max_volume() * multiplier

                # push the event to the extraction vessel
                event = ['pour by volume', beaker_2, d_volume]
                reward = extract_vessel.push_event_to_queue(events=[event], dt=self.dt)

                # push no events to the first beaker
                beaker_1.push_event_to_queue(dt=self.dt)

            # pour the (first) Solute Vessel into the Extraction Vessel
            if do_action == 5:
                # determine the volume to pour from the solute vessel into the extraction vessel
                d_volume = solute_vessel1.get_max_volume() * multiplier

                # push the event to the solute vessel
                event = ['pour by volume', extract_vessel, d_volume]
                reward = solute_vessel1.push_event_to_queue(events=[event], dt=self.dt)

                # push no events to either of the beakers
                beaker_1.push_event_to_queue(dt=self.dt)
                beaker_2.push_event_to_queue(dt=self.dt)

            # pour the (second) Solute Vessel into the Extraction Vessel
            if do_action == 6:
                # determine the volume to pour from the solute vessel into the extraction vessel
                d_volume = solute_vessel2.get_max_volume() * multiplier

                # push the event to the solute vessel
                event = ['pour by volume', extract_vessel, d_volume]
                reward = solute_vessel2.push_event_to_queue(events=[event], dt=self.dt)

                # push no events to either of the beakers
                beaker_1.push_event_to_queue(dt=self.dt)
                beaker_2.push_event_to_queue(dt=self.dt)

            # Indicate that all no more actions are to be completed
            if do_action == 7:
                # pass the fulfilled `done` parameter
                done = True

                # look through each vessel's material dict looking for the target material
                reward = 0
                all_vessels = vessels + external_vessels
                for in_vessel in all_vessels:
                    names = [name for name, __ in in_vessel._material_dict.items()]
                    if self.target_material in names:
                        reward += self.done_reward(in_vessel)

        return vessels, external_vessels, reward, done

    def done_reward(self, beaker):  # `beaker` is the beaker used to collect extracted material
        '''
        '''

        material_amount = beaker.get_material_amount(self.target_material)
        init_target_amount = self.target_material_init_amount

        if abs(material_amount - 0) < 1e-6:
            reward = -100
        else:
            try:
                assert (abs(init_target_amount - 0.0) > 1e-6)

                reward = (material_amount / init_target_amount) * 100

                print(
                    "done_reward: {}, in_beaker_2: {}, initial: {}".format(
                        reward,
                        material_amount,
                        init_target_amount
                    )
                )

            except AssertionError:
                reward = 0
                print("Oops! Division by zero. There's no target material in Extraction Vessel")

        return reward
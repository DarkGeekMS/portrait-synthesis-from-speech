import numpy as np
import math
import argparse

def get_unit_vector(vector):
    # get unit vector of a given vector
    return np.divide(vector, np.sqrt(np.dot(vector, vector)))

def disentangle_axes(source_axis, target_axis):
    # disentangle source and target axes
    # read source and target axes
    source_direction = np.load(source_axis)
    target_direction = np.load(target_axis)
    # compute and print average angle between different components (before disentanglement)
    initial_angles = list()
    for layer in range(source_direction.shape[0]):
            initial_angles.append(
                math.degrees(math.acos(np.dot(
                    get_unit_vector(source_direction[layer]), get_unit_vector(target_direction[layer])
            ))))
    print(f'Angle before disentanglement = {sum(initial_angles)/len(initial_angles)}')
    # create a new source direction with same source dimensions
    new_source_direction = np.zeros(source_direction.shape)
    # loop over each component in source direction
    for layer in range(source_direction.shape[0]):
        # project source direction along target direction
        component = np.dot(source_direction[layer], get_unit_vector(target_direction[layer]))
        diff_direction = component * get_unit_vector(target_direction[layer])
        # subtract source direction from its projection to orthogonalize source direction
        new_source_direction[layer] = source_direction[layer] - diff_direction
    # compute and print average angle between different components (after disentanglement)
    final_angles = list()
    for layer in range(new_source_direction.shape[0]):
            final_angles.append(
                math.degrees(math.acos(np.dot(
                    get_unit_vector(new_source_direction[layer]), get_unit_vector(target_direction[layer])
            ))))
    print(f'Angle after disentanglement = {sum(final_angles)/len(final_angles)}')
    # convert into unit direction
    final_source_direction = np.zeros(new_source_direction.shape)
    for layer in range(new_source_direction.shape[0]):
        final_source_direction[layer] = np.divide(
            new_source_direction[layer], np.sqrt(np.dot(new_source_direction[layer], new_source_direction[layer]))
        )
    # save new source direction
    np.save(source_axis[:-4]+'_new.npy', final_source_direction)

if __name__ == "__main__":
    # arguments parsing
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-sa', '--source_axis', type=str, help='path to feature axis to be changed')
    argparser.add_argument('-ta', '--target_axis', type=str, help='path to reference feature axis')

    args = argparser.parse_args()

    # call axis disentanglement
    disentangle_axes(args.source_axis, args.target_axis)

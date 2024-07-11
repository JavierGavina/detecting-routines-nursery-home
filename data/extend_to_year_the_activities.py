import json
import os.path

import numpy as np
import pandas as pd

from PIL import Image, ImageDraw, ImageFont

# ALL CONSTANTS FROM THE SCRIPT
PROBABILITY_OF_RANDOMNESS = 0.05
USERS = ["02A8", "9FE9", "52EA", "402E", "682A", "F176"]
DIFFICULTIES = ["easy", "medium", "hard"]
PATH_METADATA = "metadata/assigned_activities"


def read_metadata(path_to_json: str) -> dict:
    """
    This function reads the metadata from a json file.

    Parameters:
        * path_to_json (str): The path to the json file.

    Returns:
        * dict: The metadata read from the json file.

    Raises:
        FileNotFoundError: If the file does not exist.
    """

    if os.path.exists(path_to_json) is False:
        raise FileNotFoundError(f"The file {path_to_json} does not exist.")

    with open(path_to_json, 'r') as file:
        metadata = json.load(file)

    return metadata


def get_weekday(date: str) -> str:
    """
    This function returns the weekday of a given date.

    Parameters:
        * date (str): The date in the format "YYYY-MM-DD".

    Returns:
        *   str: The weekday of the given date.
    """

    year, month, day = date.split("-")

    return pd.Timestamp(year=int(year), month=int(month), day=int(float(day))).day_name()


def get_typeDates_for_each_weekday(path_to_json: str) -> dict:
    """
    This function returns a dictionary mapping each weekday to its corresponding typeDate.

    Parameters:
        path_to_json (str): The path to the json file containing the metadata.

    Returns:
        dict: A dictionary where the keys are the weekdays and the values are the typeDates.
    """

    # Read the metadata from the json file
    metadata = read_metadata(path_to_json)

    # Define a list of all weekdays
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    # Initialize an empty dictionary to store the typeDates for each weekday
    type_dates_for_each_weekday = dict()

    # Iterate over each day in the metadata
    for day in metadata:
        # Get the weekday and typeDate for the current day
        weekday = get_weekday(day["date"])
        typeDate = day["typeDate"]

        # If the weekday is not already in the dictionary, add it with its corresponding typeDate
        if weekday not in type_dates_for_each_weekday.keys():
            type_dates_for_each_weekday[weekday] = typeDate

        # If all weekdays have been added to the dictionary, break the loop
        if all([weekday in type_dates_for_each_weekday.keys() for weekday in weekdays]):
            break

    # Return the dictionary of typeDates for each weekday
    return type_dates_for_each_weekday


def extend_to_year(path_to_json: str, path_out: str, add_randomness: bool = False,
                   prob_fail: float | int = 0.05) -> None:
    """
    This function extends the metadata for a given month to a full year. It can also add randomness to the typeDates.

    Parameters:
        * path_to_json (str): The path to the json file containing the metadata.
        * path_out (str): The path where the output json file will be saved.
        * add_randomness (bool): If True, adds randomness to the typeDates. Default is False.
        * prob_fail (float | int): The probability of a typeDate failing. Default is 0.05.

    """
    # Read the metadata from the json file
    metadata = read_metadata(path_to_json)

    # Get the month from the first date in the metadata
    month_generated = metadata[0]["date"].split("-")[1]

    # Define the number of days in each month
    days_of_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    # Get the typeDates for each weekday
    type_dates_for_each_weekday = get_typeDates_for_each_weekday(path_to_json)

    # Initialize an empty list to store the output data
    out_json: list[dict] = []

    # Iterate over each month
    for month in range(1, 13):

        # If the current month is the month from the metadata, add the metadata to the output data
        if month == month_generated:
            for x in metadata:
                out_json.append(x)
        else:
            # Otherwise, generate new data for each day of the month
            for day in range(1, days_of_month[month - 1] + 1):
                # Get the weekday and typeDate for the current day
                weekday = get_weekday(f"2023-{month}-{day}")
                typedate = type_dates_for_each_weekday[weekday]

                # If add_randomness is True, possibly change the typeDate randomly
                if add_randomness:
                    if np.random.rand() < prob_fail:
                        options = [x for x in type_dates_for_each_weekday.keys() if x != weekday]
                        weekday = np.random.choice(options, 1).item()
                        typedate = type_dates_for_each_weekday[weekday]

                # Add the current day to the output data
                out_json.append({"date": f"2023-{month}-{day}", "typeDate": typedate})

    # If the output file already exists, remove it
    if os.path.exists(path_out):
        os.remove(path_out)

    # Write the output data to a json file
    with open(path_out, 'w') as file:
        json.dump(out_json, file, indent=3)


def get_colors():
    """
    This function returns a list of RGB color tuples for each location.

    Returns:
        list[tuple[int, int, int]]: A list of RGB color tuples.
    """

    l_colors = []
    l_colors.append((255, 255, 255))  # N/A: White
    l_colors.append((255, 0, 0))  # TV Room: Red
    l_colors.append((0, 255, 0))  # Living Room: Green
    l_colors.append((0, 0, 255))  # Therapy: Blue
    l_colors.append((255, 255, 0))  # Gym: Yellow
    l_colors.append((128, 0, 128))  # Corridor: Purple
    l_colors.append((0, 255, 255))  # Terrace: Cyan
    l_colors.append((255, 165, 0))  # Dinning Room: Orange
    l_colors.append((144, 238, 144))  # Room: Light Green
    l_colors.append((0, 0, 0))  # Weekend: Black

    return l_colors


def paint_pixel(pixels: any, r_start: int, c_start: int, r_end: int, c_end: int, color: tuple[int, int, int]):
    """
    This function paints a pixel in the image with the given color.

    Parameters:
        * pixels (any): The pixels of the image.
        * r_start (int): The starting row of the pixel.
        * c_start (int): The starting column of the pixel.
        * r_end (int): The ending row of the pixel.
        * c_end (int): The ending column of the pixel.
        * color (tuple[int, int, int]): The RGB color tuple.
    """

    for c in range(c_start, c_end):
        for r in range(r_start, r_end):
            pixels[c, r] = color


def add_labels(img: Image.Image, nr: int) -> Image.Image:
    """
    This function adds labels to the image.

    Parameters:
        * img (Image.Image): The image to which the labels will be added.
        * nr (int): The row number where the labels will be added.

    Returns:
        * Image.Image: The image with the labels added.
    """

    # Define the path to the font file
    font_dir = "C:\Windows\Fonts\calibri.ttf"

    # Load the font and get the pixels of the image
    font = ImageFont.truetype(font_dir, 30)
    pixels = img.load()

    # Draw a black line for each hour of the day
    for i in range(24):
        p = i * 60 * 2
        for r in range(nr - 30, nr - 20):
            pixels[p, r] = (0, 0, 0)

    # Obtain a draw object
    draw = ImageDraw.Draw(img)

    # Add text for each hour of the day
    for i in range(24):
        p = i * 60 * 2 + 5
        draw.text((p, nr - 30), str(i), (0, 0, 0), font=font)

    return img


def add_legend(img: Image.Image) -> Image.Image:
    """
    This function adds a legend to the image.

    Parameters:
        * img (Image.Image): The image to which the legend will be added.

    Returns:
        * Image.Image: The image with the legend added.
    """

    # Create blank image for the legend
    legend = Image.new('RGB', (200, 200), color=(255, 255, 255))

    # Obtain a draw object
    draw = ImageDraw.Draw(legend)

    # Define the colors and values which represents the image
    colors = get_colors()[:9]
    values = ["NA", "TV.Room", "Living.Room", "Therapy", "Gym", "Corridor", "Terrace", "Dinning.Room", "Room"]

    # Add text for each color on the legend
    font = ImageFont.truetype("C:\Windows\Fonts\calibri.ttf", 20)  # Select the font and letter size
    for i in range(len(colors)):
        draw.rectangle((10, 10 + i * 20, 30, 30 + i * 20), fill=colors[i])  # Draw a square with the color
        draw.text((40, 15 + i * 20), values[i], font=font, fill=(0, 0, 0))  # Add the name of the color

    # Combine the original image with the legend
    result = Image.new('RGB', (img.width + legend.width + 10, max(img.height, legend.height)),
                       (255, 255, 255))  # Create a big blank image
    result.paste(img, (0, 0))  # Paste the original image
    result.paste(legend, (img.width + 10, 0))

    return result


def create_image_location_map(input_data: list[list[int]], image_filename: str,
                              colors: list[tuple[int, int, int]]) -> None:
    """
    This function creates a location map image based on the input data and saves it to a file.

    Parameters:
        *  input_data (list[list[int]]): A 2D list representing the input data.
        *  image_filename (str): The name of the file where the image will be saved.
        *  colors (list[tuple[int, int, int]]): A list of RGB color tuples.
    """
    # Define the height and width of each rectangle in the image
    r_height, c_width = 10, 2

    # Calculate the total number of columns and rows in the image
    nc = 60 * 24 * c_width
    nr = len(input_data) * r_height

    # Create a new image with the calculated dimensions and white background
    img = Image.new('RGB', (nc, nr), color='white')
    pixels = img.load()

    # Initialize the starting row
    r_start = 0
    r_end = r_start + r_height

    # Iterate over each sequence in the input data
    for sq in input_data:
        # Initialize the starting column for each sequence
        c_start = 0
        c_end = c_start + c_width

        # Iterate over each item in the sequence
        for i in range(len(sq)):
            # Get the color for the current item
            color = colors[sq[i]]

            # Paint a pixel in the image with the current color
            paint_pixel(pixels, r_start, c_start, r_end, c_end, color)

            # Move to the next column
            c_start = c_start + c_width
            c_end = c_end + c_width

        # Move to the next row
        r_start = r_start + r_height
        r_end = r_start + r_height

    # Add labels to the image
    add_labels(img, nr - 3)

    # Add a legend to the image
    img = add_legend(img)

    # Save the image to a file
    img.save(image_filename)


def get_sequence_from_location_map(path_to_activity_simulation: str, padding=False, show_weekends=False) -> list[
    list[int]]:
    """
    This function reads a label map from a CSV file and returns a list of sequences.
    It also has options to add padding and show weekends.

    Parameters:
        * path_to_activity_simulation (str): The path to the label map CSV file.
        * padding (bool): Whether to add padding to the sequences.
        * show_weekends (bool): Whether to show weekends in the sequences.

    Returns:
        * list[list[int]]: A list of sequences.
    """
    # Read the CSV file, skipping the first row and without a header
    data = pd.read_csv(path_to_activity_simulation, skiprows=1, header=None)

    # Assign column names to the DataFrame
    data.columns = ["Year", "Month", "Day"] + [f"Sequence_{x}" for x in range(1439)]

    # Filter the DataFrame to only include data for the month of March
    data = data.query("Month==3")

    # Initialize an empty list to store the sequences
    list_of_sequences = []

    # Define a sequence for weekends
    weekend_sequence = np.array([[9, 9, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0, 0, 0]
                                 for _ in range(1439 // 16)]).flatten().tolist() + [9] * 8 + [0] * 7

    # Iterate over each row in the DataFrame
    for row in data.iterrows():
        # Construct the date string
        date = f"2023-3-{row[1][2]}"

        # Get the sequence from the row and convert any NaN values to 0
        sequence = row[1][3:].to_list()
        sequence = [int(x) if not np.isnan(x) else 0 for x in sequence]

        # Add the sequence to the list of sequences
        list_of_sequences.append(sequence)

        # If show_weekends is True and the date is a Sunday, add the weekend sequence
        if show_weekends:
            if get_weekday(date) == "Sunday":
                list_of_sequences.append(weekend_sequence)

    # If padding is True, add padding to the list of sequences
    if padding:
        for i in range(4):
            list_of_sequences.append([0] * 1439)

    # Return the list of sequences
    return list_of_sequences


if __name__ == "__main__":

    # Extend the activities to simulate a year
    for user in USERS:
        path = f"{user}/{PATH_METADATA}"
        extend_to_year(path_to_json=f"{path}.json", path_out=f"{path}_year.json")

        # Add randomness to the activities to simulate a year on hard mode
        extend_to_year(path_to_json=f"{path}.json", path_out=f"{path}_year_random.json",
                       add_randomness=True, prob_fail=PROBABILITY_OF_RANDOMNESS)
        print(f"Processed {user}")

    # Plot the groundtruth location map for the original month (March)
    for user in USERS:
        for dificulty in DIFFICULTIES:
            path_data = f"{user}/{dificulty}/activities-simulation.csv"
            sequences = get_sequence_from_location_map(path_to_activity_simulation=path_data, padding=True,
                                                       show_weekends=True)

            create_image_location_map(input_data=sequences, image_filename=f"{user}/{dificulty}/groundtruth.png",
                                      colors=get_colors())

    input("Press Any Key to End... ")

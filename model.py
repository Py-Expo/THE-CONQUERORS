import cv2
import os
import openpyxl

# Function to detect color of a car
def detect_car_color(image):
    # Convert image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define color ranges for different car colors
    color_ranges = {
        'red': ((0, 100, 100), (10, 255, 255)),
        'blue': ((110, 50, 50), (130, 255, 255)),
        'green': ((36, 25, 25), (86, 255, 255)),
        'yellow': ((20, 100, 100), (30, 255, 255)),
        'white': ((0, 0, 200), (180, 40, 255)),
        'black': ((0, 0, 0), (180, 255, 30))
    }

    # Iterate through color ranges and find the color of the car
    max_area = 0
    car_color = 'unknown'
    for color, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(hsv_image, lower, upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        area = 0
        for contour in contours:
            area += cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            car_color = color

    print("Detected Car Color:", car_color)  # Print detected color
    return car_color

# Function to perform number plate recognition
def recognize_number_plates(frame, save_path, excel_file):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply image processing techniques to isolate number plate regions
    
    # Example: Apply thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Example: Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Initialize a counter for the file name
    file_counter = 0
    
    # Initialize an Excel workbook and worksheet
    workbook = openpyxl.Workbook()
    worksheet = workbook.active
    worksheet.append(["File Name", "Car Color"])
    
    # Iterate through contours
    for contour in contours:
        # Get bounding box for each contour
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filter contours based on aspect ratio and area
        aspect_ratio = w / float(h)
        if 2.5 < aspect_ratio < 4 and cv2.contourArea(contour) > 1000:
            # Extract number plate region
            number_plate = frame[y:y+h, x:x+w]
            
            # Detect car color
            car_color = detect_car_color(number_plate)
            
            # Save the number plate image
            filename = os.path.join(save_path, f'number_plate_{file_counter}.jpg')
            cv2.imwrite(filename, number_plate)
            
            # Increment the file counter
            file_counter += 1

            # Display number plate region
            cv2.imshow('Number Plate', number_plate)
            
            # Write car color to Excel
            worksheet.append([filename, car_color])

    # Save Excel file
    excel_file_name = os.path.join(save_path, excel_file)
    workbook.save(excel_file_name)
    print("Excel file saved at:", excel_file_name)  # Print Excel file path

def main():
    # Open video capture device (0 is usually the default webcam)
    cap = cv2.VideoCapture(0)

    save_path = "number_plate_images"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    excel_file = "car_colors.xlsx"

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret:
            # Display the captured frame
            cv2.imshow('Live Video', frame)

            # Perform number plate recognition and save the images
            recognize_number_plates(frame, save_path, excel_file)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

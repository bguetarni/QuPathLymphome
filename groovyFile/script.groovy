/* groovylint-disable JavaIoPackageAccess, LineLength */
import java.awt.image.BufferedImage
import java.awt.BorderLayout
import java.nio.file.Files
import java.nio.file.Paths
import javax.imageio.ImageIO
import javax.swing.JFileChooser
import javax.swing.JFrame
import javax.swing.JList
import javax.swing.JScrollPane
import javax.swing.JOptionPane
import javax.swing.JButton
import javax.swing.ListSelectionModel

import com.google.gson.JsonArray
import com.google.gson.JsonElement
import com.google.gson.JsonObject
import com.google.gson.JsonParser

import qupath.lib.images.servers.openslide.OpenslideImageServer
import qupath.lib.objects.PathObjects
import qupath.lib.roi.ROIs
import qupath.lib.regions.ImagePlane
import qupath.lib.objects.classes.PathClassFactory


// ==============   SCRIPT CONSTANTS   =================

PYTHON_ENV_NAME = "pythonEnv"
SCRIPT_NAME = "launchApp.py"

TASKS = ["molecular subtyping": "subtyping", "treatment response": "treatment"]

DISPLAY_HEATMAPS = false

DISPLAY_GENERAL_INFORMATION = true

DOWNSAMPLE_FACTOR_PNG = 1.5

DASH_URL = "127.0.0.1"
DASH_PORT = "8050"

// ====================================================



// ============   PYTHON ERROR CODES   ================

PYTHON_ERROR = [
    3: "WSI file was not found.",
    4: "Annotation file was not found",
]

// ====================================================


private static String chooseDirectory(String message) {
    JFileChooser chooser = new JFileChooser()
    chooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY)
    chooser.setDialogTitle(message)
    if (chooser.showOpenDialog(null) == JFileChooser.APPROVE_OPTION) {
        File selectedFile = chooser.getSelectedFile()
        return selectedFile.getAbsolutePath()
    }
    else {
        return null
    }
}

def displayHeatMapOnQupath(imageFolder) {
    selectObjects { p -> (p.getLevel() == 1) && (p.isAnnotation() == false) };
    clearSelectedObjects(false)

    // Utiliser Files pour parcourir tous les fichiers du répertoire
    Files.walk(Paths.get(imageFolder)).forEach { filePath ->
    
    // Vérifier si le fichier est un fichier JSON
    if (Files.isRegularFile(filePath) && filePath.toString().toLowerCase().endsWith("result.json")) {
        try {
            // Lire le contenu du fichier JSON
            String jsonString = new String(Files.readAllBytes(filePath))
            
            // Parser le contenu JSON et poursuivre comme avant
            JsonElement element = JsonParser.parseString(jsonString)
            JsonObject jsonObject = element.getAsJsonObject()
            JsonArray tilesArray = jsonObject.getAsJsonArray('tiles')
 
            // Définir le plan et les classes de chemin
            int z = 0
            int t = 0
            ImagePlane plane = ImagePlane.getPlane(z, t)
 
            // Liste pour stocker les détections
            var detections = []
 
            // Itérer sur chaque élément du tableau JSON et créer des objets de détection
            for (JsonElement tileElement : tilesArray) {
                JsonObject tileObject = tileElement.getAsJsonObject()

                // Extraire x, y, largeur et hauteur
                double x = tileObject.get('xmin').getAsDouble()
                double y = tileObject.get('ymin').getAsDouble()
                double probability = tileObject.get('lymphome_probability').getAsDouble()
                double width = Math.abs(tileObject.get('xmax').getAsDouble() - x)
                double height = Math.abs(tileObject.get('ymax').getAsDouble() - y)
 
                // Créer une ROI basée sur les coordonnées du tile
                var roi = ROIs.createRectangleROI(x, y, width, height, plane)
 
                // Créer un objet de détection basé sur la probabilité
                var pathclass = PathClass.getInstance('predictor+')
                var detection = PathObjects.createDetectionObject(roi, pathclass)
                double red = (170 * probability) + 20
                double green = (170 * (1 - probability)) + 20
                double blue = 40
 
                if (probability >= 0 && probability <= 1) {
                    detection.setColor((int)red, (int)green, (int)blue)
                }
                else {
                    detection.setColor(255, 255, 255)
                }
 
                // Ajouter l'objet de détection à la liste
                detections.add(detection)
            }
 
            // Ajouter tous les objets de détection au projet QuPath
            addObjects(detections)
            } catch (Exception e) {
                println("Erreur lors de la lecture du fichier: " + filePath.toString())
                e.printStackTrace()
            }
        }
    }
}

String selectTask() {
    def frame = new JFrame("Select an Option")
    def list = new JList(TASKS.keySet()  as Object[])
    list.setSelectionMode(ListSelectionModel.SINGLE_SELECTION)
    def scrollPane = new JScrollPane(list)
    def button = new JButton("Select")
    def selectedValue = null

    button.addActionListener {
        selectedValue = list.getSelectedValue()
        frame.dispose()
    }

    frame.setLayout(new BorderLayout())
    frame.add(scrollPane, BorderLayout.CENTER)
    frame.add(button, BorderLayout.SOUTH)
    frame.setSize(300, 200)
    frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE)
    frame.setVisible(true)

    // Wait for the user to select an option
    while (frame.isVisible()) {
        Thread.sleep(100)
    }
    
    String task = TASKS.get(selectedValue)

    return task
}

// Check image is opened
ImageData imageData = QP.getCurrentImageData()
if (imageData == null) {
    JOptionPane.showMessageDialog(null, "No image is opened. Please open an image first.",
    "Error no image", JOptionPane.ERROR_MESSAGE)
    return -1
}

// general message
if (DISPLAY_GENERAL_INFORMATION) {
    JOptionPane.showMessageDialog(null,
    "This is a general message information about the usage.\n\n \
In this tool, two options are available: \n \
     DLBCL subtyping \n \
     DLBCL treatment response prediction \n\n \
Manual annotations have to be selected in order to perform the task.",
    "General information", JOptionPane.INFORMATION_MESSAGE)
}

// Select the task to perform
String task = selectTask()
if (task == null) {
    JOptionPane.showMessageDialog(null, "No task was selected. Exiting.",
    "Error no task", JOptionPane.ERROR_MESSAGE)
    return -1
}

// Check at least one annotation is selected
var rois = selectedObjects
if (rois.size() == 0) {
    JOptionPane.showMessageDialog(null, "No annotation is selected. Please select an annotation and give it a name.",
    "Error of selected annotation", JOptionPane.ERROR_MESSAGE)
    return -1
}
else if (rois.size() > 1) {
    JOptionPane.showMessageDialog(null, "More than 1 annotation was detected, please only select one annotation.",
    "Error of selected annotation", JOptionPane.ERROR_MESSAGE)
    return -1
}

// Optional output path (can be removed)
String pathOutput = chooseDirectory("Choose the QupathLymphoma folder")
if (pathOutput == null) {
    return -1
}

// Remove the file extension ".tif"
String imageName = GeneralTools.getNameWithoutExtension(imageData.getServer().getMetadata().getName())
// alternative:    imageName = imageName.substring(0, imageName.lastIndexOf('.'))

// Construct the full path for the image folder
String imageFolder = buildFilePath(pathOutput, "blendmaps2", "data", imageName)

// Check if the image folder exists, create it if not
if (!new File(imageFolder).exists()) {
    new File(imageFolder).mkdirs()
}

// We only work with one annotation for now
// TBD later: handle multiple annotations at once
roi = rois[0]

// Get the annotation name using getName() method
String annotationName = roi.getName()

// Create the annotation folder path within wsi folder
if (annotationName == null) {
    int resp = JOptionPane.showConfirmDialog(null, "The annotation doesn't have a name. Do you want to use the object ID as name ?")
    if (resp == JOptionPane.YES_OPTION) {
        annotationName = roi.getID().toString()
    }
    else {
        annotationName = JOptionPane.showInputDialog("Please type in the name for the annotation.")
        if (annotationName == null || annotationName  == "") {
            JOptionPane.showMessageDialog(null, "No name was typed, exiting.", "Name empty", JOptionPane.ERROR_MESSAGE)
            return -1
        }
    }
}

String annotationFolderPath = buildFilePath(imageFolder, annotationName)

// Check if annotation folder exists, create if not
if (!new File(annotationFolderPath).exists()) {
    new File(annotationFolderPath).mkdirs()
}

// Build the final JSON file path within the annotation folder
String fileOutput = buildFilePath(annotationFolderPath, annotationName + ".json")

// Write (save) the JSON file
var gson = GsonTools.getInstance(true)
try (Writer writer = new FileWriter(fileOutput)) {
    gson.toJson(roi, writer)
} catch (IOException e) {
    e.printStackTrace()
    JOptionPane.showMessageDialog(null, "An error occured during the writing of the JSON file.",
    "Error", JOptionPane.ERROR_MESSAGE)
    return -1
}

// PARTIE IMAGE
// Chargez l'image à partir du chemin
var originalImage = new OpenslideImageServer(QP.getCurrentImageData().getServer().getURIs()[0])

// Obtenez la région sélectionnée (ROI)
   var region = roi.getROI()
  
// Obtenez les coordonnées de la région
int  x = region.getBoundsX()
int  y = region.getBoundsY()
int width = region.getBoundsWidth()
int height = region.getBoundsHeight()

// Découpez la région de l'image originale
BufferedImage regionImage = originalImage.readRegion(DOWNSAMPLE_FACTOR_PNG, x, y, width, height)

// Construisez le chemin de sortie pour l'image PNG
String imageOutputPath = buildFilePath(annotationFolderPath, annotationName + ".png")

// Enregistrez l'image découpée au format PNG
ImageIO.write(regionImage, "png", new File(imageOutputPath))

if (DISPLAY_HEATMAPS) {
    int res = JOptionPane.showOptionDialog(new JFrame(), "Do you want to display previous heatmap on this image ?", "Heatmap",
    JOptionPane.YES_NO_OPTION, JOptionPane.QUESTION_MESSAGE, null,
    new Object[] { 'Yes', 'No' }, JOptionPane.YES_OPTION)

    if (res == JOptionPane.YES_OPTION) {
        displayHeatMapOnQupath(imageFolder)
    }    
}

try {
    String pythonExecutable = buildFilePath(pathOutput, PYTHON_ENV_NAME, "Scripts", "python.exe")
    print(pythonExecutable)
    if (pythonExecutable == null) {
        JOptionPane.showMessageDialog(null, "The python.Exe file cannot be found. Please re-execute the setup script.",
        "Error", JOptionPane.ERROR_MESSAGE)
        return -1
    }
    
    String pythonScript = buildFilePath(pathOutput, "blendmaps2", SCRIPT_NAME)
    
    String path = QP.getCurrentImageData().getServer().getURIs()[0].getPath().substring(1)
    
    // create python command-line
    ProcessBuilder processBuilder = new ProcessBuilder(pythonExecutable, pythonScript, 
    '--outputPath', imageFolder, 
    '--wsiPath', path,
    "--task", task,
    "--url", DASH_URL,
    "--port", DASH_PORT)
    
    // launch python script
    processBuilder.inheritIO()
    Process process = processBuilder.start()
    print("Executing python script...")
    int exitCode = process.waitFor()
    
    // display potential error message
    print("Python script exited with code " + exitCode)
    if(PYTHON_ERROR.containsKey(exitCode)) {
        print(PYTHON_ERROR[exitCode])
    }
} catch (IOException | InterruptedException e) {
    JOptionPane.showMessageDialog(null, "An exceptio occured during the attempt to launch the python code.",
    "Exception error", JOptionPane.ERROR_MESSAGE)
    e.printStackTrace()
    return -1
}

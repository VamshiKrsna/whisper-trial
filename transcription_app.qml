import QtQuick 2.15
import QtQuick.Controls 2.15

ApplicationWindow {
    visible: true
    width: 640
    height: 480
    title: "Whisper Transcription App"

    Column {
        anchors.centerIn: parent
        spacing: 10

        Text {
            id: transcriptionDisplay
            objectName: "transcriptionDisplay"  // Set objectName for Python access
            text: "Transcription will appear here..."
            font.pixelSize: 20
        }

        Button {
            id: startButton
            objectName: "startButton"  // Set objectName for Python access
            text: "Start Transcription"
        }

        Button {
            id: stopButton
            objectName: "stopButton"  // Set objectName for Python access
            text: "Stop Transcription"
        }
    }
}
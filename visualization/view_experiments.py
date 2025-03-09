from qtpy.QtWidgets import QApplication

from c_elegans_utils.visualization.experiment_viewer import ExperimentsViewer

# You need one (and only one) QApplication instance per application.
app = QApplication([])

# Create a Qt widget, which will be our window.
window = ExperimentsViewer()
window.show()  # IMPORTANT!!!!! Windows are hidden by default.

# Start the event loop.
app.exec()


# Your application won't reach here until you exit and the event
# loop has stopped.

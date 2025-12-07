# Heart Rate GUI Project — Design Iteration Summary

This project evolved from a basic real-time PPG visualizer into a more structured,
user-centered heart-rate monitoring interface. Early iterations focused on
achieving reliable connectivity and waveform streaming. Later iterations refined
the UI based on usability and cognitive design principles.

---

## Iteration 1 → Final GUI Improvements

| Area | Iteration 1 | Final GUI |
|------|-------------|-----------|
| **Layout** | Sparse right-panel BPM display; controls loosely placed | Clear 3-region structure: header controls → waveform → stats/notes |
| **Controls** | Only Connect/Disconnect | Added Refresh, Record, Snapshot, port/baud selection |
| **User Feedback** | Minimal inline text | Dedicated status bar: connection + recording indicators |
| **Visual Hierarchy** | Users scan entire window for state cues | Waveform emphasized; controls grouped by function |
| **Cognitive Support** | Limited spatial organization | Proximity, structure, and color improve recognition and scanning time |
| **Session Context** | No integrated notes or export | Notes panel + CSV export pathway for portability |

---

## Summary

The final design shifted from *proof-of-connection* to a usable monitoring tool with:

- Better grouping of interactive controls
- Clearer feedback about system state
- Support for note-taking and future data analysis workflows

This scope evolution ensured timely delivery by a single developer while laying a solid
foundation for future enhancements like wearable hardware and multi-profile data hosting.

---

### Tools and References

- **GUI**: [Qt Widgets](https://doc.qt.io/qt-6/qtwidgets-index.html)
- **Charting**: [PyQtGraph](https://pyqtgraph.readthedocs.io/en/latest/)
- **Computation**: [NumPy](https://numpy.org/doc/stable/)

Development support assisted by **GPT-5.1** and **Claude Sonnet 4.5**.

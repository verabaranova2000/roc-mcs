Option Explicit
Dim fso, shell, root, app, pyw, candidate
Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")
root = fso.GetParentFolderName(WScript.ScriptFullName)
app = fso.BuildPath(root, "src\roc_mcs\gui\live_scans\live_scans_app.py")
pyw = "pythonw.exe"
For Each candidate In Array(fso.BuildPath(root, ".venv\Scripts\pythonw.exe"), fso.BuildPath(root, "venv\Scripts\pythonw.exe"))
  If fso.FileExists(candidate) Then
    pyw = candidate
    Exit For
  End If
Next
If Not fso.FileExists(app) Then
  MsgBox "Не найден файл запуска:" & vbCrLf & app, vbCritical, "Live ROC Dashboard"
Else
  shell.Run Chr(34) & pyw & Chr(34) & " " & Chr(34) & app & Chr(34), 0, False
End If

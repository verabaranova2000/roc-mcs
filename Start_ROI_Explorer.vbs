Option Explicit
Dim fso, shell, root, app, pyw, cfg, ts, candidate
Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")
root = fso.GetParentFolderName(WScript.ScriptFullName)
app = fso.BuildPath(root, "src\roc_mcs\gui\roi_explorer\roi_explorer_app.py")
pyw = "pythonw.exe"
For Each candidate In Array(fso.BuildPath(root, ".venv\Scripts\pythonw.exe"), fso.BuildPath(root, "venv\Scripts\pythonw.exe"))
  If fso.FileExists(candidate) Then pyw = candidate: Exit For
Next
cfg = fso.BuildPath(root, "pythonw_path.txt")
If fso.FileExists(cfg) Then
  Set ts = fso.OpenTextFile(cfg, 1, False)
  Do Until ts.AtEndOfStream
    candidate = Trim(ts.ReadLine)
    If Len(candidate) > 0 And Left(candidate, 1) <> "#" Then
      If fso.FileExists(candidate) Then pyw = candidate
      Exit Do
    End If
  Loop
  ts.Close
End If
If Not fso.FileExists(app) Then
  MsgBox "Не найден файл запуска:" & vbCrLf & app, vbCritical, "ROI Explorer"
Else
  On Error Resume Next
  shell.Run Chr(34) & pyw & Chr(34) & " " & Chr(34) & app & Chr(34), 0, False
  If Err.Number <> 0 Then MsgBox "Не удалось запустить Python:" & vbCrLf & pyw & vbCrLf & "Проверь pythonw_path.txt.", vbCritical, "ROI Explorer"
  On Error GoTo 0
End If

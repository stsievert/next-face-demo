
## NEXT face embedding demo

Start the demo with `bokeh serve myapp`.

### TODOs
1. make the feature finding of the face local
2. use webcam instead of file upload

This will mean that

1. make the feature finding of the face local
    * Images will not have to uploaded to dropbox
    * Face++ API will not be used
    * embedding new images will be much quicker (no upload)

2. [ ] use webcam instead of file upload
    * can have live updates, not discrete updates
    * might change threat model?

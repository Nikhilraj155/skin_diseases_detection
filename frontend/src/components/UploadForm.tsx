import { ChangeEvent } from "react";

type UploadFormProps = {
  previewUrl: string | null;
  isSubmitting: boolean;
  onFileChange: (file: File | null) => void;
  onSubmit: () => void;
};

export function UploadForm({
  previewUrl,
  isSubmitting,
  onFileChange,
  onSubmit,
}: UploadFormProps) {
  function handleChange(event: ChangeEvent<HTMLInputElement>) {
    const nextFile = event.target.files?.[0] ?? null;
    onFileChange(nextFile);
  }

  return (
    <section className="panel upload-panel">
      <div>
        <p className="eyebrow">Upload</p>
        <h2>Analyze your skin image</h2>
        <p className="muted">
          Upload a clear photo of the affected skin area. Our system will analyze the image 
          and provide detailed information about the condition.
        </p>
      </div>

      <label className="upload-dropzone">
        <input type="file" accept="image/*" onChange={handleChange} />
        {previewUrl ? (
          <img src={previewUrl} alt="Selected preview" className="preview-image" />
        ) : (
          <span>Select an image file to analyze</span>
        )}
      </label>

      <button className="primary-button" type="button" disabled={isSubmitting} onClick={onSubmit}>
        {isSubmitting ? "Analyzing..." : "Analyze Image"}
      </button>
    </section>
  );
}

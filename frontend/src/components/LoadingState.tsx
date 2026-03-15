export function LoadingState() {
  return (
    <div className="loading-state" aria-live="polite">
      <div className="loading-orb" />
      <span>Running model inference and building the response...</span>
    </div>
  );
}

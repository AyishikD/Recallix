/* Product films rendered from a real Recallix session (source: recallix-video/, outside this repo). */
export function DemoVideo({ src, poster, label, autoPlay = false, className = "" }:
    { src: string; poster: string; label: string; autoPlay?: boolean; className?: string }) {
    return (
        <div className={`overflow-hidden rounded-[28px] card-edge bg-code ${className}`}>
            <video
                className="block w-full h-auto"
                src={src}
                poster={poster}
                aria-label={label}
                controls
                playsInline
                preload={autoPlay ? "auto" : "none"}
                {...(autoPlay ? { autoPlay: true, muted: true, loop: true } : {})}
            />
        </div>
    );
}

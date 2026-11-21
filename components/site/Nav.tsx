"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";

export const NAV_LINKS = [
    { href: "/flow-control", label: "How it works" },
    { href: "/architecture", label: "Architecture" },
    { href: "/benchmark", label: "Benchmarks" },
    { href: "/sdk", label: "Developers" },
    { href: "/roadmap", label: "Roadmap" },
    { href: "/observability", label: "Dashboard" },
];

function Logo() {
    return (
        <Link href="/" className="flex items-center gap-2 text-ink" aria-label="Recallix home">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                <circle cx="12" cy="12" r="10.5" stroke="currentColor" strokeWidth="1.6" />
                <circle cx="12" cy="12" r="4.5" fill="currentColor" />
            </svg>
            <span className="text-[15px] font-semibold tracking-tight">Recallix</span>
        </Link>
    );
}

export function Nav() {
    const pathname = usePathname();
    const [open, setOpen] = useState(false);
    return (
        <header className="sticky top-0 z-50 nav-glass border-b border-black/[0.06]">
            <nav className="mx-auto flex h-12 max-w-[1024px] items-center justify-between px-5">
                <Logo />
                <ul className="hidden md:flex items-center gap-7">
                    {NAV_LINKS.map(l => (
                        <li key={l.href}>
                            <Link href={l.href}
                                className={`text-[12px] transition-colors ${pathname === l.href ? "text-ink" : "text-ink/70 hover:text-ink"}`}>
                                {l.label}
                            </Link>
                        </li>
                    ))}
                </ul>
                <Link href="/sdk" className="hidden md:inline-flex h-7 items-center rounded-full bg-accent px-3 text-[12px] font-medium text-white hover:bg-accent-hover transition-colors">
                    Get started
                </Link>
                <button className="md:hidden p-2 -mr-2 text-ink" aria-label="Menu" aria-expanded={open} onClick={() => setOpen(!open)}>
                    <svg width="18" height="18" viewBox="0 0 18 18" aria-hidden="true">
                        {open
                            ? <path d="M4 4l10 10M14 4L4 14" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
                            : <path d="M2 6h14M2 12h14" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />}
                    </svg>
                </button>
            </nav>
            {open && (
                <ul className="md:hidden border-t border-black/[0.06] px-5 pb-4">
                    {NAV_LINKS.map(l => (
                        <li key={l.href}>
                            <Link href={l.href} onClick={() => setOpen(false)}
                                className="block py-3 text-[17px] font-semibold text-ink border-b border-black/[0.06]">
                                {l.label}
                            </Link>
                        </li>
                    ))}
                </ul>
            )}
        </header>
    );
}

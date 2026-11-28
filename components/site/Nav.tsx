"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";

import { NAV_LINKS } from "./links";

export { NAV_LINKS };

function Logo() {
    return (
        <Link href="/" className="flex items-center gap-2 text-ink" aria-label="Recallic home">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                <path d="M11.2 4.6C9.6 3.6 7.2 4 6.4 5.8C4.6 6 3.4 7.8 4 9.6C2.8 10.6 2.8 12.8 4.1 13.8C3.6 15.8 5 17.6 6.8 17.6C7.4 19.4 9.8 20.2 11.2 19ZM12.8 4.6C14.4 3.6 16.8 4 17.6 5.8C19.4 6 20.6 7.8 20 9.6C21.2 10.6 21.2 12.8 19.9 13.8C20.4 15.8 19 17.6 17.2 17.6C16.6 19.4 14.2 20.2 12.8 19Z"
                    stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
                <path d="M6.4 9.4C7.6 8.8 8.8 9.8 9.8 9.2M5.8 13.2C7.2 13.8 8.4 12.8 9.8 13.6M7.8 16.2C8.6 15.6 9.4 15.8 10 16.4M17.6 9.4C16.4 8.8 15.2 9.8 14.2 9.2M18.2 13.2C16.8 13.8 15.6 12.8 14.2 13.6M16.2 16.2C15.4 15.6 14.6 15.8 14 16.4"
                    stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" />
                <circle cx="12" cy="2.6" r="2" fill="#FF6B5B" />
            </svg>
            <span className="text-[15px] font-semibold tracking-tight">Recallic</span>
        </Link>
    );
}

export function Nav() {
    const pathname = usePathname();
    const [open, setOpen] = useState(false);
    return (
        <header className="sticky top-0 z-50 nav-glass border-b border-white/[0.07]">
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
                <Link href="/sdk" className="hidden md:inline-flex h-7 items-center rounded-full bg-cta px-3 text-[12px] font-semibold text-cta-ink hover:bg-cta-hover transition-colors">
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
                <ul className="md:hidden border-t border-white/[0.07] px-5 pb-4">
                    {NAV_LINKS.map(l => (
                        <li key={l.href}>
                            <Link href={l.href} onClick={() => setOpen(false)}
                                className="block py-3 text-[17px] font-semibold text-ink border-b border-white/[0.07]">
                                {l.label}
                            </Link>
                        </li>
                    ))}
                </ul>
            )}
        </header>
    );
}

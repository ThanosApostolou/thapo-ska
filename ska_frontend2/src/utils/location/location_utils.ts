export class LocationUtils {
    static getHostname(): string {
        return window.location.hostname;
    }

    static getHref(): string {
        return window.location.href;
    }

    static getPathname(): string {
        return window.location.pathname;
    }

    static getPort(): string {
        return window.location.port;
    }

    static reload() {
        window.location.reload();
    }

    static setHref(href: string) {
        window.location.href = href;
    }

    static replace(url: string | URL) {
        window.location.replace(url);
    }

}
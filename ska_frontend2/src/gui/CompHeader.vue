<script setup lang="ts">
import { GlobalState } from '@/domain/global_state/global_state';

const globalState = GlobalState.instance();

</script>

<template>
  <header class="navbar bg-neutral shadow-lg">
    <div class="flex flex-row flex-1">
      <label for="my-drawer" class="btn drawer-button">
        <img src="/assets/icons/bors-3.svg" width="24" />
      </label>
      <RouterLink :to="globalState.appRoutes.PAGE_HOME" class="btn btn-ghost text-neutral-content text-xl">
        Specific Knowledge Assistant
      </RouterLink>

      <span class="flex-1"></span>
      <details class="dropdown dropdown-end">
        <summary class="m-1 btn">
          <!-- <Show when=move || global_store.get().refresh_token.get().is_some()>
            <p>
              { move ||
              match &global_store.get().user_details.get() {
              Some(user_details) => user_details.username.clone(),
              None => "unknown".to_string()
              }
              }
            </p>
          </Show> -->
          <img src="/assets/icons/user-circle.svg" width="24" />
          <img src="/assets/icons/chevron-down.svg" width="16" />
        </summary>
        <ul class="p-2 shadow menu dropdown-content z-[1] bg-base-100 rounded-box w-32">
          <li>
            <!-- <button class="btn btn-ghost" on:click=move |_| { login_action.dispatch(()) }>
              <img src="/assets/icons/arrow-right-start-on-rectangle.svg" width="24" />login
            </button> -->
          </li>
          <li>
            <!-- <button class="btn btn-ghost" on:click=move |_| { logout_action.dispatch(()) }>
              <img src="/assets/icons/arrow-right-start-on-rectangle.svg" width="24" />logout
            </button> -->
          </li>
        </ul>
      </details>
    </div>
  </header>
</template>
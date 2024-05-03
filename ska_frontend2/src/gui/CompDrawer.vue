<script setup lang="ts">
import { ref } from 'vue';
import CompFooter from './CompFooter.vue';
import CompHeader from './CompHeader.vue';
import { GlobalState } from '@/domain/global_state/global_state';

const globalState = GlobalState.instance();

// hooks
const isChecked = ref(false);

</script>

<template>
  <div class="drawer w-full h-full flex flex-col items-stretch">
    <input id="my-drawer" type="checkbox" class="drawer-toggle" v-model="isChecked" />
    <div class="drawer-side">
      <label for="my-drawer" aria-label="close sidebar" class="drawer-overlay"></label>
      <ul class="menu p-4 w-80 min-h-full bg-base-200 text-base-content">
        <li>
          <RouterLink @click="() => isChecked = false" :to="globalState.appRoutes.PAGE_HOME" activeClass="active">
            <img src="/assets/icons/home.svg" width="24" />Home
          </RouterLink>
        </li>
        <li>
          <RouterLink @click="() => isChecked = false" :to="globalState.appRoutes.PAGE_ASSISTANT" activeClass="active">
            <img src="/assets/icons/chat-bubble-left-right.svg" width="24" />Assistant
          </RouterLink>
        </li>
      </ul>
    </div>

    <div class="drawer-content w-full h-full flex flex-col items-stretch">
      <div class="flex-none">
        <CompHeader />
      </div>

      <!-- <Routes base=base_href.with_untracked(move |base_href| base_href.clone())>
        //
        <AppRoute />
        <Route path="/home" view=PageHome />
        <Route path="/assistant" view=PageAssistant />
        <Route path="/account" view=PageAccount />
        <Route path="/login" view=PageLogin />
        <Route path="" view=move || { view! { <Redirect path="home" /> }} />
        <Route path="*" view=PageNotFound />
      </Routes> -->
      <RouterView />

      <div class="flex-none">
        <CompFooter />
      </div>

    </div>
  </div>

</template>